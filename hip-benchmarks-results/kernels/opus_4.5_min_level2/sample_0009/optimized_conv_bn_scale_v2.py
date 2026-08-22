import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Improved fused BatchNorm + Scaling kernel with better memory coalescing
# Uses channel-wise processing for better cache utilization

fused_bn_scale_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Process one channel worth of data per thread block
// Better memory coalescing and cache utilization
__global__ void fused_bn_scale_kernel_optimized(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ fused_scale,
    const float* __restrict__ fused_bias,
    int N, int C, int HW) {
    
    // Each block handles one (n, c) pair
    int nc_idx = blockIdx.x;
    int n = nc_idx / C;
    int c = nc_idx % C;
    
    if (n >= N) return;
    
    // Load scale and bias for this channel into registers (shared across all threads in block)
    float scale = fused_scale[c];
    float bias = fused_bias[c];
    
    // Base offset for this (n, c) slice
    int base = (n * C + c) * HW;
    
    // Process 4 elements per thread when possible
    int hw4 = HW / 4;
    int remaining = HW % 4;
    
    // Process vectorized elements
    for (int i = threadIdx.x; i < hw4; i += blockDim.x) {
        int offset = base + i * 4;
        float4 x = *reinterpret_cast<const float4*>(input + offset);
        float4 y;
        y.x = x.x * scale + bias;
        y.y = x.y * scale + bias;
        y.z = x.z * scale + bias;
        y.w = x.w * scale + bias;
        *reinterpret_cast<float4*>(output + offset) = y;
    }
    
    // Handle remaining elements
    int start_remaining = hw4 * 4;
    for (int i = threadIdx.x; i < remaining; i += blockDim.x) {
        int offset = base + start_remaining + i;
        output[offset] = input[offset] * scale + bias;
    }
}

// Alternative kernel using grid-strided loop with vectorization
__global__ void fused_bn_scale_kernel_grid_stride(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ fused_scale,
    const float* __restrict__ fused_bias,
    int N, int C, int HW, int total) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process 4 elements at a time
    int total4 = total / 4;
    
    for (int i = idx; i < total4; i += stride) {
        int base_idx = i * 4;
        
        // Calculate channel for first element
        int c = (base_idx / HW) % C;
        
        float scale = fused_scale[c];
        float bias = fused_bias[c];
        
        float4 x = *reinterpret_cast<const float4*>(input + base_idx);
        float4 y;
        y.x = x.x * scale + bias;
        y.y = x.y * scale + bias;
        y.z = x.z * scale + bias;
        y.w = x.w * scale + bias;
        *reinterpret_cast<float4*>(output + base_idx) = y;
    }
    
    // Handle remaining elements
    int start = total4 * 4;
    for (int i = start + idx; i < total; i += stride) {
        int c = (i / HW) % C;
        output[i] = input[i] * fused_scale[c] + fused_bias[c];
    }
}

torch::Tensor fused_bn_scale_hip(
    torch::Tensor input,
    torch::Tensor fused_scale,
    torch::Tensor fused_bias) {
    
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int HW = H * W;
    
    auto output = torch::empty_like(input);
    
    // Use the optimized kernel that processes per (n, c) pair
    int num_nc_pairs = N * C;
    const int block_size = 256;
    
    fused_bn_scale_kernel_optimized<<<num_nc_pairs, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        fused_scale.data_ptr<float>(),
        fused_bias.data_ptr<float>(),
        N, C, HW);
    
    return output;
}
"""

fused_bn_scale_cpp = """
torch::Tensor fused_bn_scale_hip(torch::Tensor input, torch::Tensor fused_scale, torch::Tensor fused_bias);
"""

fused_bn_scale = load_inline(
    name="fused_bn_scale",
    cpp_sources=fused_bn_scale_cpp,
    cuda_sources=fused_bn_scale_source,
    functions=["fused_bn_scale_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs convolution, then fused BatchNorm + Scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor
        self.fused_bn_scale = fused_bn_scale
        
        # Precomputed fused parameters
        self.register_buffer('fused_scale', None)
        self.register_buffer('fused_bias', None)
        
    def _compute_fused_params(self):
        """Compute fused scale and bias from BN parameters"""
        with torch.no_grad():
            gamma = self.bn.weight
            beta = self.bn.bias
            mean = self.bn.running_mean
            var = self.bn.running_var
            eps = self.bn.eps
            
            inv_std = 1.0 / torch.sqrt(var + eps)
            self.fused_scale = (gamma * inv_std * self.scaling_factor).contiguous()
            self.fused_bias = ((beta - mean * gamma * inv_std) * self.scaling_factor).contiguous()

    def forward(self, x):
        x = self.conv(x)
        
        if self.fused_scale is None or self.training:
            self._compute_fused_params()
        
        x = self.fused_bn_scale.fused_bn_scale_hip(x, self.fused_scale, self.fused_bias)
        
        return x


def get_inputs():
    return [torch.rand(128, 8, 128, 128).cuda()]


def get_init_inputs():
    return [8, 64, 3, 2.0]
