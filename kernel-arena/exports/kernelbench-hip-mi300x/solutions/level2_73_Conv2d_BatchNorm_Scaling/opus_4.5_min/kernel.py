import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused scaling kernel - just multiply by a constant per channel
# This is very simple because we fold BN into conv weights
fused_scale_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Fuse BN into conv weights and just do scaling at the end
__global__ void channel_scale_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float scaling_factor,
    const int total) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total) {
        output[idx] = input[idx] * scaling_factor;
    }
}

// Vectorized scaling kernel
__global__ void channel_scale_kernel_vec4(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    const float scaling_factor,
    const int total4) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total4) {
        float4 x = input[idx];
        float4 y;
        y.x = x.x * scaling_factor;
        y.y = x.y * scaling_factor;
        y.z = x.z * scaling_factor;
        y.w = x.w * scaling_factor;
        output[idx] = y;
    }
}

torch::Tensor channel_scale_hip(torch::Tensor input, float scaling_factor) {
    int total = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    
    if (total % 4 == 0) {
        int total4 = total / 4;
        int num_blocks = (total4 + block_size - 1) / block_size;
        channel_scale_kernel_vec4<<<num_blocks, block_size>>>(
            reinterpret_cast<const float4*>(input.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            scaling_factor,
            total4);
    } else {
        int num_blocks = (total + block_size - 1) / block_size;
        channel_scale_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            scaling_factor,
            total);
    }
    
    return output;
}
"""

fused_scale_cpp = """
torch::Tensor channel_scale_hip(torch::Tensor input, float scaling_factor);
"""

fused_scale = load_inline(
    name="fused_scale",
    cpp_sources=fused_scale_cpp,
    cuda_sources=fused_scale_source,
    functions=["channel_scale_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


def fuse_conv_bn_weights(conv_weight, conv_bias, bn_weight, bn_bias, bn_mean, bn_var, bn_eps):
    """Fuse conv + BN weights into a single conv layer."""
    # BN: y = (x - mean) / sqrt(var + eps) * gamma + beta
    # Conv + BN: y = conv(x) * (gamma / sqrt(var + eps)) + (beta - mean * gamma / sqrt(var + eps) + conv_bias * gamma / sqrt(var + eps))
    
    inv_std = 1.0 / torch.sqrt(bn_var + bn_eps)
    
    # New conv weight: w' = w * gamma / sqrt(var + eps)
    # Weight shape: [out_channels, in_channels, H, W]
    fused_weight = conv_weight * (bn_weight * inv_std).view(-1, 1, 1, 1)
    
    # New conv bias: b' = (b - mean) * gamma / sqrt(var + eps) + beta
    if conv_bias is not None:
        fused_bias = (conv_bias - bn_mean) * bn_weight * inv_std + bn_bias
    else:
        fused_bias = (-bn_mean) * bn_weight * inv_std + bn_bias
    
    return fused_weight, fused_bias


class ModelNew(nn.Module):
    """
    Optimized model: Fuse BatchNorm into Conv weights, then apply scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        # Keep original layers for weight initialization
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor
        self.fused_scale = fused_scale
        
        # Buffer for fused conv (created lazily)
        self.register_buffer('fused_conv_weight', None)
        self.register_buffer('fused_conv_bias', None)
        
    def _fuse_weights(self):
        """Fuse BN into conv weights."""
        with torch.no_grad():
            fused_weight, fused_bias = fuse_conv_bn_weights(
                self.conv.weight,
                self.conv.bias,
                self.bn.weight,
                self.bn.bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.eps
            )
            self.fused_conv_weight = fused_weight.contiguous()
            self.fused_conv_bias = fused_bias.contiguous()

    def forward(self, x):
        # Fuse weights if not done yet (only for inference)
        if self.fused_conv_weight is None or self.training:
            self._fuse_weights()
        
        # Apply fused conv (Conv + BN combined)
        x = F.conv2d(x, self.fused_conv_weight, self.fused_conv_bias)
        
        # Apply scaling
        x = self.fused_scale.channel_scale_hip(x, self.scaling_factor)
        
        return x


def get_inputs():
    return [torch.rand(128, 8, 128, 128).cuda()]


def get_init_inputs():
    return [8, 64, 3, 2.0]
