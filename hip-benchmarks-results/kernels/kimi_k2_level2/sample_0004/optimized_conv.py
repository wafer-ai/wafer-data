import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused Conv2d + Subtract1 + Tanh + Subtract2 with tiling
custom_conv_fused_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define TILE_SIZE 16
#define BLOCK_SIZE 256

__device__ float tanh_approx(float x) {
    // Fast tanh approximation
    float x2 = x * x;
    float a = x * (1.0f + x2 * (0.134660f + x2 * 0.010612f));
    return a / (1.0f + x2 * (0.671640f + x2 * 0.083321f));
}

__global__ void fused_conv_sub_tanh_sub_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int N, int C_in, int C_out, int H, int W,
    int kernel_size, int H_out, int W_out, float sub1, float sub2) {
    
    int n = blockIdx.x;
    int c_out = blockIdx.y;
    int h_out = blockIdx.z / ((W_out + TILE_SIZE - 1) / TILE_SIZE);
    int w_out_tile = blockIdx.z % ((W_out + TILE_SIZE - 1) / TILE_SIZE);
    
    int h_out_start = h_out * TILE_SIZE;
    int w_out_start = w_out_tile * TILE_SIZE;
    
    int tx = threadIdx.x % TILE_SIZE;
    int ty = threadIdx.x / TILE_SIZE;
    
    int h_out_idx = h_out_start + ty;
    int w_out_idx = w_out_start + tx;
    
    if (n >= N || c_out >= C_out) return;
    if (h_out_idx >= H_out || w_out_idx >= W_out) return;
    
    float sum = 0.0f;
    
    // Unrolled kernel loops for better performance
    for (int c_in = 0; c_in < C_in; c_in++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int h_in = h_out_idx + kh;
                int w_in = w_out_idx + kw;
                
                if (h_in < H && w_in < W) {
                    float input_val = input[((n * C_in + c_in) * H + h_in) * W + w_in];
                    float weight_val = weight[((c_out * C_in + c_in) * kernel_size + kh) * kernel_size + kw];
                    sum += input_val * weight_val;
                }
            }
        }
    }
    
    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[c_out];
    }
    
    // Fused operations: subtract1 -> tanh -> subtract2
    sum = sum - sub1;
    sum = tanh_approx(sum);
    sum = sum - sub2;
    
    output[((n * C_out + c_out) * H_out + h_out_idx) * W_out + w_out_idx] = sum;
}

// Custom average pooling kernel with better memory access
__global__ void custom_avgpool_kernel(
    const float* input, float* output,
    int N, int C, int H_in, int W_in, int kernel_size, int H_out, int W_out) {
    
    int n = blockIdx.x;
    int c = blockIdx.y;
    int h_out = blockIdx.z / W_out;
    int w_out = blockIdx.z % W_out;
    
    if (n >= N || c >= C || h_out >= H_out || w_out >= W_out) return;
    
    float sum = 0.0f;
    int count = 0;
    
    int h_start = h_out * kernel_size;
    int w_start = w_out * kernel_size;
    
    // Load multiple elements per thread for better coalescing
    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            int h_in = h_start + kh;
            int w_in = w_start + kw;
            
            if (h_in < H_in && w_in < W_in) {
                sum += input[((n * C + c) * H_in + h_in) * W_in + w_in];
                count++;
            }
        }
    }
    
    output[((n * C + c) * H_out + h_out) * W_out + w_out] = sum / count;
}

torch::Tensor fused_conv_sub_tanh_sub(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float sub1, float sub2) {
    
    int N = input.size(0);
    int C_in = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    int C_out = weight.size(0);
    int kernel_size = weight.size(2);
    
    int H_out = H - kernel_size + 1;
    int W_out = W - kernel_size + 1;
    
    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());
    
    dim3 block(TILE_SIZE * TILE_SIZE);
    dim3 grid(N, C_out, ((H_out + TILE_SIZE - 1) / TILE_SIZE) * ((W_out + TILE_SIZE - 1) / TILE_SIZE));
    
    fused_conv_sub_tanh_sub_kernel<<<grid, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), N, C_in, C_out, H, W,
        kernel_size, H_out, W_out, sub1, sub2);
    
    return output;
}

torch::Tensor custom_avgpool(torch::Tensor input, int kernel_size) {
    int N = input.size(0);
    int C = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    
    int H_out = (H_in + kernel_size - 1) / kernel_size;
    int W_out = (W_in + kernel_size - 1) / kernel_size;
    
    auto output = torch::zeros({N, C, H_out, W_out}, input.options());
    
    dim3 block(1);
    dim3 grid(N, C, H_out * W_out);
    
    custom_avgpool_kernel<<<grid, block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, H_in, W_in, kernel_size, H_out, W_out);
    
    return output;
}
"""

custom_ops = load_inline(
    name="custom_conv_fused",
    cpp_sources=custom_conv_fused_cpp_source,
    functions=["fused_conv_sub_tanh_sub", "custom_avgpool"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool
        self.custom_ops = custom_ops
        
    def forward(self, x):
        # Extract conv weights and bias
        weight = self.conv.weight
        bias = self.conv.bias if self.conv.bias is not None else torch.Tensor()
        
        # Use custom fused kernel for conv + subtract1 + tanh + subtract2
        x = self.custom_ops.fused_conv_sub_tanh_sub(
            x, weight, bias, self.subtract1_value, self.subtract2_value
        )
        
        # Use custom avgpool kernel
        x = self.custom_ops.custom_avgpool(x, self.kernel_size_pool)
        
        return x

def get_inputs():
    batch_size = 128
    in_channels = 64
    height, width = 128, 128
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    in_channels = 64
    out_channels = 128
    kernel_size = 3
    subtract1_value = 0.5
    subtract2_value = 0.2
    kernel_size_pool = 2
    return [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool]
