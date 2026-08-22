import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused BatchNorm + Scaling kernel with better thread utilization
fused_bn_scale_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Kernel that processes multiple (N,C) pairs per block for better occupancy
__global__ void fused_bn_scale_kernel_v5(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ fused_scale,
    const float* __restrict__ fused_bias,
    const int N, const int C, const int HW) {
    
    // Process one (n, c) pair per block
    int nc_idx = blockIdx.x;
    int n = nc_idx / C;
    int c = nc_idx % C;
    
    if (n >= N) return;
    
    float scale = fused_scale[c];
    float bias = fused_bias[c];
    
    int base = (n * C + c) * HW;
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    
    // Use float4 for vectorized access
    int hw4 = HW / 4;
    
    const float4* input4 = reinterpret_cast<const float4*>(input + base);
    float4* output4 = reinterpret_cast<float4*>(output + base);
    
    for (int i = tid; i < hw4; i += total_threads) {
        float4 x = input4[i];
        float4 y;
        y.x = fmaf(x.x, scale, bias);
        y.y = fmaf(x.y, scale, bias);
        y.z = fmaf(x.z, scale, bias);
        y.w = fmaf(x.w, scale, bias);
        output4[i] = y;
    }
    
    // Handle remaining
    int start_remaining = hw4 * 4;
    for (int i = tid + start_remaining; i < HW; i += total_threads) {
        output[base + i] = fmaf(input[base + i], scale, bias);
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
    
    int num_nc_pairs = N * C;
    int block_size = 256;
    
    // Clamp block size based on HW
    if (HW < 256) block_size = 128;
    if (HW < 128) block_size = 64;
    
    fused_bn_scale_kernel_v5<<<num_nc_pairs, block_size>>>(
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
    Optimized model with fused Conv+BN+Scale.
    During inference, we fuse BN parameters into the conv weights.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor
        self.fused_bn_scale = fused_bn_scale
        
        self.register_buffer('fused_scale', None)
        self.register_buffer('fused_bias', None)
        self._fused = False
        
    def _compute_fused_params(self):
        with torch.no_grad():
            gamma = self.bn.weight
            beta = self.bn.bias
            mean = self.bn.running_mean
            var = self.bn.running_var
            eps = self.bn.eps
            
            inv_std = torch.rsqrt(var + eps)
            self.fused_scale = (gamma * inv_std * self.scaling_factor).contiguous()
            self.fused_bias = ((beta - mean * gamma * inv_std) * self.scaling_factor).contiguous()

    def forward(self, x):
        # First do convolution
        x = self.conv(x)
        
        # Compute fused BN+scale params if not yet done
        if self.fused_scale is None or self.training:
            self._compute_fused_params()
        
        # Apply fused BN + scaling
        x = self.fused_bn_scale.fused_bn_scale_hip(x, self.fused_scale, self.fused_bias)
        
        return x


def get_inputs():
    return [torch.rand(128, 8, 128, 128).cuda()]


def get_init_inputs():
    return [8, 64, 3, 2.0]
