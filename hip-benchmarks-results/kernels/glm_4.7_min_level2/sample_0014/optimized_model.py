import os
import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused kernel
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <math.h>

__global__ void fused_linear_gelu_softmax_kernel(
    const float* input,      // [batch_size, in_features]
    const float* weight,     // [out_features, in_features]
    const float* bias,       // [out_features]
    float* output,           // [batch_size, out_features]
    int batch_size,
    int in_features,
    int out_features) {
    
    int batch_idx = blockIdx.x;
    int out_idx = threadIdx.x;
    
    if (out_idx >= out_features) return;
    
    // Start with bias
    float sum = bias[out_idx];
    
    // Vector dot product
    for (int in_idx = 0; in_idx < in_features; in_idx++) {
        sum += input[batch_idx * in_features + in_idx] * weight[out_idx * in_features + in_idx];
    }
    
    // Apply GELU
    float x = sum;
    float gelu_val = x * 0.5f * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    
    // Softmax across features
    __shared__ float smem[256];
    int tid = out_idx;
    
    smem[tid] = (tid < out_features) ? gelu_val : 0.0f;
    __syncthreads();
    
    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s && (tid + s) < out_features) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }
    __syncthreads();
    max_val = smem[0];
    __syncthreads();
    
    // Compute exp and sum
    float exp_val = (tid < out_features) ? expf(gelu_val - max_val) : 0.0f;
    
    smem[tid] = exp_val;
    __syncthreads();
    
    // Reduction for sum
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s && (tid + s) < out_features) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    __syncthreads();
    
    float sum_exp = smem[0];
    
    // Output
    int global_idx = batch_idx * out_features + out_idx;
    if (tid < out_features) {
        output[global_idx] = exp_val / sum_exp;
    }
}
"""

fused_linear_gelu_softmax = load_inline(
    name="fused_lgs_simple",
    cpp_sources=fused_kernel_source,
    functions=["fused_linear_gelu_softmax_kernel"],
    verbose=False,
    with_cuda=True,
)


class ModelNew(nn.Module):
    """Optimized model with fused Matmul+GELU+Softmax"""
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
        self.fused_kernel = fused_linear_gelu_softmax

    def forward(self, x):
        batch_size = x.size(0)
        output = torch.zeros(batch_size, self.out_features, device=x.device, dtype=x.dtype)
        
        block_size = min(256, self.out_features)
        num_blocks = batch_size
        
        self.fused_kernel.fused_linear_gelu_softmax_kernel(
            num_blocks, block_size, 0,
            x.data_ptr(),
            self.weight.data_ptr(),
            self.bias.data_ptr(),
            output.data_ptr(),
            batch_size,
            self.in_features,
            self.out_features
        )
        
        return output