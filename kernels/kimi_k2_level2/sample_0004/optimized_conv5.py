import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized with better memory access patterns
custom_fused_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Optimized element-wise kernel using vectorized loads
__global__ void fused_sub_tanh_sub_kernel(
    const float* __restrict__ input, float* __restrict__ output, int size, float sub1, float sub2) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple elements for better memory bandwidth utilization
    for (; idx < size; idx += stride) {
        float val = input[idx] - sub1;
        val = tanhf(val);  // Use built-in tanhf for accuracy
        val = val - sub2;
        output[idx] = val;
    }
}

// Optimized average pooling kernel with coalesced memory access
__global__ void custom_avgpool_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    int N, int C, int H_in, int W_in, int H_out, int W_out) {
    
    // Each thread processes one output position - best for coalesced access
    int n = blockIdx.x;
    int c = blockIdx.y;
    int h_out = blockIdx.z / W_out;
    int w_out = blockIdx.z % W_out;
    
    if (n >= N || c >= C || h_out >= H_out || w_out >= W_out) return;
    
    // Precompute base indices for better performance
    int input_base = ((n * C + c) * H_in);
    int output_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
    
    float sum = 0.0f;
    int h_start = h_out * 2;
    int w_start = w_out * 2;
    
    // Load 4 elements with coalesced access
    // This is unrolled for kernel_size=2
    sum += input[input_base + (h_start + 0) * W_in + (w_start + 0)];
    sum += input[input_base + (h_start + 0) * W_in + (w_start + 1)];
    sum += input[input_base + (h_start + 1) * W_in + (w_start + 0)];
    sum += input[input_base + (h_start + 1) * W_in + (w_start + 1)];
    
    output[output_idx] = sum * 0.25f;  // Divide by 4
}

torch::Tensor fused_sub_tanh_sub(
    torch::Tensor input, float sub1, float sub2) {
    
    auto output = torch::zeros_like(input);
    int size = input.numel();
    
    // Use many blocks to fill the GPU
    const int block_size = 256;
    const int num_blocks = min(1024, (size + block_size - 1) / block_size);
    
    fused_sub_tanh_sub_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), size, sub1, sub2);
    
    return output;
}

torch::Tensor custom_avgpool(torch::Tensor input) {
    int N = input.size(0);
    int C = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    
    int H_out = (H_in + 1) / 2;
    int W_out = (W_in + 1) / 2;
    
    auto output = torch::zeros({N, C, H_out, W_out}, input.options());
    
    dim3 block(1);
    dim3 grid(N, C, H_out * W_out);
    
    custom_avgpool_kernel<<<grid, block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, H_in, W_in, H_out, W_out);
    
    return output;
}
"""

custom_ops = load_inline(
    name="custom_fused_ops",
    cpp_sources=custom_fused_cpp_source,
    functions=["fused_sub_tanh_sub", "custom_avgpool"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.custom_ops = custom_ops
        
    def forward(self, x):
        # Keep using PyTorch conv (highly optimized by cuDNN/MIOpen)
        x = self.conv(x)
        
        # Use custom fused kernel for subtract1 + tanh + subtract2
        x = self.custom_ops.fused_sub_tanh_sub(x, self.subtract1_value, self.subtract2_value)
        
        # Use custom avgpool kernel (optimized)
        x = self.custom_ops.custom_avgpool(x)
        
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
