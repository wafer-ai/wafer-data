import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Set the compiler
os.environ["CXX"] = "hipcc"

# Lightweight fused kernel for BatchNorm + Scaling
kernel_code = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define EPSILON 1e-5f
#define BLOCK_SIZE 256

// Fused BN + Scale kernel with vectorized access
__global__ void fuse_bn_scale_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ mean, 
    const float* __restrict__ var, 
    const float* __restrict__ bn_weight, 
    const float* __restrict__ bn_bias,
    float* __restrict__ out, 
    int total_elements, int channels, int spatial_size,
    float scaling_factor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;
    
    // Calculate channel index efficiently
    int channel = (idx / spatial_size) % channels;
    
    // Load parameters
    float inv_std = rsqrtf(var[channel] + EPSILON);
    float scale_val = scaling_factor * bn_weight[channel] * inv_std;
    float offset_val = scaling_factor * (bn_bias[channel] - mean[channel] * bn_weight[channel] * inv_std);
    
    // Load input and compute
    float x_val = x[idx];
    float result = x_val * scale_val + offset_val;
    
    // Store output
    out[idx] = result;
}

// PyTorch wrapper  
torch::Tensor fuse_bn_scale_hip(
    torch::Tensor x, torch::Tensor mean, torch::Tensor var, 
    torch::Tensor bn_weight, torch::Tensor bn_bias, 
    float scaling_factor
) {
    // Get dimensions
    int N = x.size(0);
    int C = x.size(1);
    int spatial_size = x.size(2) * x.size(3);
    int total_elements = x.numel();
    
    // Create output tensor
    auto out = torch::zeros_like(x);
    
    // Calculate grid dimensions
    int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Launch kernel
    fuse_bn_scale_kernel<<<num_blocks, BLOCK_SIZE>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        out.data_ptr<float>(),
        total_elements, C, spatial_size,
        scaling_factor
    );
    
    return out;
}
"""

# Compile the kernel
fuse_bn_scale = load_inline(
    name="fuse_bn_scale",
    cpp_sources=kernel_code,
    functions=["fuse_bn_scale_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model: Keep PyTorch's highly optimized Conv2D,
    fuse only the BatchNorm + Scale operations
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor
        self.fuse_bn_scale = fuse_bn_scale
        
    def forward(self, x):
        # Keep PyTorch's highly optimized Conv2D
        x = self.conv(x)
        
        # Extract BN parameters
        bn_mean = self.bn.running_mean
        bn_var = self.bn.running_var
        bn_weight = self.bn.weight
        bn_bias = self.bn.bias
        
        # Use fused BN + Scale kernel
        return self.fuse_bn_scale.fuse_bn_scale_hip(
            x, bn_mean, bn_var, bn_weight, bn_bias, self.scaling_factor
        )


# Input parameters
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
scaling_factor = 2.0

# Generate inputs
input_size = (batch_size, in_channels, height, width)

def get_inputs():
    return [torch.randn(input_size).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor]