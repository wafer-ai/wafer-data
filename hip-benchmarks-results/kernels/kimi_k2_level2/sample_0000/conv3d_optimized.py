import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define TILE_SIZE 4
#define MAX_CHANNELS 16

__global__ void fused_conv3d_softmax_maxpool_kernel(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width,
    int ksize, int pool_ksize, int stride, int pad)
{
    int out_c = blockIdx.x;
    int ow = threadIdx.x;
    int oh = threadIdx.y;
    int od = threadIdx.z;
    
    int batch_idx = blockIdx.z / out_depth;
    int od_global = blockIdx.z % out_depth;
    
    if (batch_idx >= batch_size || out_c >= out_channels || 
        ow >= out_width || oh >= out_height || od >= out_depth) return;
    
    // Compute conv3d
    float sum = 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < ksize; ++kd) {
            for (int kh = 0; kh < ksize; ++kh) {
                for (int kw = 0; kw < ksize; ++kw) {
                    int id = od_global + kd;
                    int ih = oh + kh;
                    int iw = ow + kw;
                    
                    if (id < in_depth && ih < in_height && iw < in_width) {
                        int input_idx = ((batch_idx * in_channels + ic) * in_depth + id) * 
                                      in_height * in_width + ih * in_width + iw;
                        int weight_idx = ((out_c * in_channels + ic) * ksize + kd) * 
                                       ksize * ksize + kh * ksize + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Add bias
    if (bias != nullptr) {
        sum += bias[out_c];
    }
    
    // Store conv result in shared memory for softmax
    __shared__ float conv_results[MAX_CHANNELS][TILE_SIZE][TILE_SIZE][TILE_SIZE];
    conv_results[out_c][od][oh][ow] = sum;
    __syncthreads();
    
    // Softmax: compute max for numerical stability (only first thread in block)
    if (od == 0 && oh == 0 && ow == 0) {
        for (int d = 0; d < out_depth; ++d) {
            for (int h = 0; h < out_height; ++h) {
                for (int w = 0; w < out_width; ++w) {
                    float max_val = -1e20f;
                    for (int c = 0; c < out_channels; ++c) {
                        max_val = fmaxf(max_val, conv_results[c][d][h][w]);
                    }
                    
                    // Compute exp and sum
                    float exp_sum = 0.0f;
                    for (int c = 0; c < out_channels; ++c) {
                        float exp_val = expf(conv_results[c][d][h][w] - max_val);
                        conv_results[c][d][h][w] = exp_val;
                        exp_sum += exp_val;
                    }
                    
                    // Normalize
                    for (int c = 0; c < out_channels; ++c) {
                        conv_results[c][d][h][w] /= exp_sum;
                    }
                }
            }
        }
    }
    __syncthreads();
    
    // Apply max pooling twice (effectively pool size 4)
    // First max pooling
    __shared__ float pool1_results[MAX_CHANNELS][TILE_SIZE/2][TILE_SIZE/2][TILE_SIZE/2];
    
    if (od % 2 == 0 && oh % 2 == 0 && ow % 2 == 0) {
        int pool1_od = od / 2;
        int pool1_oh = oh / 2;
        int pool1_ow = ow / 2;
        
        float max_val = -1e20f;
        for (int pd = 0; pd < 2 && od + pd < out_depth; ++pd) {
            for (int ph = 0; ph < 2 && oh + ph < out_height; ++ph) {
                for (int pw = 0; pw < 2 && ow + pw < out_width; ++pw) {
                    max_val = fmaxf(max_val, conv_results[out_c][od + pd][oh + ph][ow + pw]);
                }
            }
        }
        pool1_results[out_c][pool1_od][pool1_oh][pool1_ow] = max_val;
    }
    __syncthreads();
    
    // Second max pooling and write output
    if (od % 4 == 0 && oh % 4 == 0 && ow % 4 == 0) {
        int pool2_od = od / 4;
        int pool2_oh = oh / 4;
        int pool2_ow = ow / 4;
        
        int pool1_depth = (out_depth + 1) / 2;
        int pool1_height = (out_height + 1) / 2;
        int pool1_width = (out_width + 1) / 2;
        
        int pool1_od_base = od / 2;
        int pool1_oh_base = oh / 2;
        int pool1_ow_base = ow / 2;
        
        float max_val = -1e20f;
        for (int pd = 0; pd < 2 && pool1_od_base + pd < pool1_depth; ++pd) {
            for (int ph = 0; ph < 2 && pool1_oh_base + ph < pool1_height; ++ph) {
                for (int pw = 0; pw < 2 && pool1_ow_base + pw < pool1_width; ++pw) {
                    max_val = fmaxf(max_val, 
                        pool1_results[out_c][pool1_od_base + pd][pool1_oh_base + ph][pool1_ow_base + pw]);
                }
            }
        }
        
        int final_depth = (out_depth + 3) / 4;
        int final_height = (out_height + 3) / 4;
        int final_width = (out_width + 3) / 4;
        
        if (pool2_od < final_depth && pool2_oh < final_height && pool2_ow < final_width) {
            int output_idx = ((batch_idx * out_channels + out_c) * final_depth + pool2_od) * 
                           final_height * final_width + pool2_oh * final_width + pool2_ow;
            output[output_idx] = max_val;
        }
    }
}

torch::Tensor fused_conv3d_softmax_maxpool(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    
    int out_channels = weight.size(0);
    int ksize = weight.size(2);
    int stride = 1;
    int pad = 0;
    
    int out_depth = in_depth - ksize + 1;
    int out_height = in_height - ksize + 1;
    int out_width = in_width - ksize + 1;
    
    int final_depth = (out_depth + 3) / 4;
    int final_height = (out_height + 3) / 4;
    int final_width = (out_width + 3) / 4;
    
    auto output = torch::zeros({batch_size, out_channels, final_depth, final_height, final_width},
                              input.options());
    
    dim3 threads(TILE_SIZE, TILE_SIZE, TILE_SIZE);
    dim3 blocks(out_channels, 
                (out_height + TILE_SIZE - 1) / TILE_SIZE,
                batch_size * out_depth);
    
    fused_conv3d_softmax_maxpool_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        out_depth, out_height, out_width,
        ksize, pool_ksize, stride, pad);
    
    return output;
}
"""

fused_conv3d = load_inline(
    name="fused_conv3d",
    cpp_sources=cpp_source,
    functions=["fused_conv3d_softmax_maxpool"],
    extra_cuda_cflags=["-O3"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=True)
        self.fused_conv3d = fused_conv3d
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        # Fused conv3d + softmax + maxpool
        return self.fused_conv3d.fused_conv3d_softmax_maxpool(x, self.conv.weight, self.conv.bias)

def get_inputs():
    batch_size = 128
    in_channels = 3
    depth, height, width = 16, 32, 32
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    pool_kernel_size = 2
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
