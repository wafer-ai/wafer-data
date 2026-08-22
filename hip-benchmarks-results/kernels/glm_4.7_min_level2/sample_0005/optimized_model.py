import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused kernel with faster softplus and cleaner memory access
activation_bnorm_cpp_source = """
#include <hip/hip_runtime.h>
#include <math.h>

__device__ __forceinline__ float softplus_fast(float x) {
    // Faster approximation for softplus
    // For large x: softplus(x) ≈ x
    // For small x: softplus(x) ≈ exp(x)
    const float THRESH = 20.0f;
    
    if (x > THRESH) {
        return x;
    } else if (x < -THRESH) {
        return expf(x);
    } else {
        return log1pf(expf(x));  // log1p is more accurate than log(1+x)
    }
}

__global__ void fused_activation_bnorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int total_elements,
    int elements_per_channel,
    int channels,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;
    
    // Compute channel for this element
    int channel = (idx % (channels * elements_per_channel)) / elements_per_channel;
    
    // Load parameters
    float mean = running_mean[channel];
    float var = running_var[channel];
    float gamma = weight[channel];
    float beta = bias[channel];
    
    // Pre-compute batch norm scaling to avoid repeated division
    float inv_std = rsqrtf(var + eps);
    float scale = gamma * inv_std;
    float offset = beta - mean * scale;
    
    // Load and compute activation: x * tanh(softplus(x))
    float x = input[idx];
    float sp = softplus_fast(x);
    float activated = x * tanhf(sp);
    
    // Apply batch norm and write
    output[idx] = activated * scale + offset;
}

torch::Tensor fused_activation_bnorm_hip(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps) {
    
    auto out = torch::empty_like(input);
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    int total_elements = batch_size * channels * height * width;
    int elements_per_channel = height * width;
    
    const int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_activation_bnorm_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        total_elements,
        elements_per_channel,
        channels,
        eps);
    
    return out;
}
"""

fused_activation_bnorm = load_inline(
    name="fused_activation_bnorm",
    cpp_sources=activation_bnorm_cpp_source,
    functions=["fused_activation_bnorm_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses activation and batch normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # Keep PyTorch's optimized conv2d
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        
        # Create BatchNorm to get trained parameters
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        
        # Register our custom kernel
        self.fused_activation_bnorm = fused_activation_bnorm

    def forward(self, x):
        # Apply convolution (using PyTorch's optimized implementation)
        x = self.conv(x)
        
        # Get batchnorm parameters
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        weight = self.bn.weight
        bias = self.bn.bias
        eps = self.bn.eps
        
        # Apply fused activation + batchnorm kernel
        x = self.fused_activation_bnorm.fused_activation_bnorm_hip(
            x, running_mean, running_var, weight, bias, eps)
        
        return x