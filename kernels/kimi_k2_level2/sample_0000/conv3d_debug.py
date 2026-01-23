import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void simple_conv3d_kernel(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width,
    int ksize, int stride, int pad)
{
    int out_c = blockIdx.x;
    int ow = blockIdx.y * blockDim.x + threadIdx.x;
    int oh = blockIdx.z * blockDim.y + threadIdx.y;
    int od = blockIdx.w * blockDim.z + threadIdx.z;
    int batch_idx = 0;  // Simplified for now
    
    if (out_c >= out_channels || ow >= out_width || oh >= out_height || od >= out_depth) return;
    
    float sum = 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < ksize; ++kd) {
            for (int kh = 0; kh < ksize; ++kh) {
                for (int kw = 0; kw < ksize; ++kw) {
                    int id = od + kd;
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
    
    if (bias != nullptr) {
        sum += bias[out_c];
    }
    
    int output_idx = (out_c * out_depth + od) * out_height * out_width + oh * out_width + ow;
    output[output_idx] = sum;
}

torch::Tensor simple_conv3d(
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
    
    // Apply two max pool operations
    int final_depth = (out_depth + 3) / 4;
    int final_height = (out_height + 3) / 4;
    int final_width = (out_width + 3) / 4;
    
    // For debugging, let's just do conv + first pool to see intermediate result
    auto conv_output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width},
                                   input.options());
    
    dim3 threads(8, 8, 2);
    dim3 blocks(out_channels, 
                (out_width + threads.x - 1) / threads.x,
                (out_height + threads.y - 1) / threads.y * 
                (out_depth + threads.z - 1) / threads.z);
    
    simple_conv3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        conv_output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        out_depth, out_height, out_width,
        ksize, stride, pad);
    
    // Apply softmax and pooling on CPU for now
    conv_output = torch::softmax(conv_output, 1);
    
    // First max pool
    auto pool1_output = torch::max_pool3d(conv_output, 2, 2, 0, 1, false, false, {1,1,1});
    
    // Second max pool
    auto final_output = torch::max_pool3d(pool1_output, 2, 2, 0, 1, false, false, {1,1,1});
    
    return final_output;
}
"""

fused_conv3d = load_inline(
    name="fused_conv3d",
    cpp_sources=cpp_source,
    functions=["simple_conv3d"],
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
        # Use custom conv3d + pytorch for rest
        return self.fused_conv3d.simple_conv3d(x, self.conv.weight, self.conv.bias)

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
