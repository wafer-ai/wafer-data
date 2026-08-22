import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused BatchNorm + Scaling kernel
# During inference, BN is: y = (x - mean) / sqrt(var + eps) * gamma + beta
# With scaling: y = ((x - mean) / sqrt(var + eps) * gamma + beta) * scale
# Can be rewritten as: y = x * fused_scale + fused_bias
# where: fused_scale = gamma * scale / sqrt(var + eps)
#        fused_bias = (beta - mean * gamma / sqrt(var + eps)) * scale

fused_bn_scale_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void fused_bn_scale_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ fused_scale,
    const float* __restrict__ fused_bias,
    int N, int C, int H, int W) {
    
    int total = N * C * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total) {
        // Calculate channel index
        int hw = H * W;
        int c = (idx / hw) % C;
        
        float x = input[idx];
        output[idx] = x * fused_scale[c] + fused_bias[c];
    }
}

// Vectorized version for better memory throughput
__global__ void fused_bn_scale_kernel_vec4(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    const float* __restrict__ fused_scale,
    const float* __restrict__ fused_bias,
    int N, int C, int H, int W) {
    
    int hw = H * W;
    int total_vec4 = (N * C * hw) / 4;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_vec4) {
        // Calculate which elements we're processing
        int base_idx = idx * 4;
        int c = (base_idx / hw) % C;
        
        float4 x = input[idx];
        float scale = fused_scale[c];
        float bias = fused_bias[c];
        
        float4 result;
        result.x = x.x * scale + bias;
        result.y = x.y * scale + bias;
        result.z = x.z * scale + bias;
        result.w = x.w * scale + bias;
        
        output[idx] = result;
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
    
    auto output = torch::empty_like(input);
    
    int total = N * C * H * W;
    const int block_size = 256;
    
    // Use vectorized kernel if possible (when H*W is divisible by 4)
    if ((H * W) % 4 == 0 && total >= 4) {
        int total_vec4 = total / 4;
        int num_blocks = (total_vec4 + block_size - 1) / block_size;
        
        fused_bn_scale_kernel_vec4<<<num_blocks, block_size>>>(
            reinterpret_cast<const float4*>(input.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            fused_scale.data_ptr<float>(),
            fused_bias.data_ptr<float>(),
            N, C, H, W);
    } else {
        int num_blocks = (total + block_size - 1) / block_size;
        
        fused_bn_scale_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            fused_scale.data_ptr<float>(),
            fused_bias.data_ptr<float>(),
            N, C, H, W);
    }
    
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
        
        # Precomputed fused parameters (will be set during first forward or after loading weights)
        self.register_buffer('fused_scale', None)
        self.register_buffer('fused_bias', None)
        
    def _compute_fused_params(self):
        """Compute fused scale and bias from BN parameters"""
        with torch.no_grad():
            # Get BN parameters
            gamma = self.bn.weight
            beta = self.bn.bias
            mean = self.bn.running_mean
            var = self.bn.running_var
            eps = self.bn.eps
            
            # Compute fused parameters
            # fused_scale = gamma * scaling_factor / sqrt(var + eps)
            # fused_bias = (beta - mean * gamma / sqrt(var + eps)) * scaling_factor
            inv_std = 1.0 / torch.sqrt(var + eps)
            self.fused_scale = (gamma * inv_std * self.scaling_factor).contiguous()
            self.fused_bias = ((beta - mean * gamma * inv_std) * self.scaling_factor).contiguous()

    def forward(self, x):
        # Apply convolution
        x = self.conv(x)
        
        # Recompute fused params if needed (handles weight changes during training)
        if self.fused_scale is None or self.training:
            self._compute_fused_params()
        
        # Apply fused BN + scaling
        x = self.fused_bn_scale.fused_bn_scale_hip(x, self.fused_scale, self.fused_bias)
        
        return x


def get_inputs():
    return [torch.rand(128, 8, 128, 128).cuda()]


def get_init_inputs():
    return [8, 64, 3, 2.0]
