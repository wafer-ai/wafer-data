import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused Mish + BatchNorm kernel for inference
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

// Fast mish with numerical stability using fast math intrinsics
__device__ __forceinline__ float fast_mish(float x) {
    float sp;
    if (x > 20.0f) {
        sp = x;
    } else if (x < -20.0f) {
        sp = __expf(x);
    } else {
        sp = __logf(1.0f + __expf(x));
    }
    return x * tanhf(sp);
}

// Fused Mish + BatchNorm kernel
// For each channel c: output = gamma[c] * (mish(input) - mean[c]) / sqrt(var[c] + eps) + beta[c]
// But during inference with running stats: output = gamma[c] / sqrt(running_var[c] + eps) * mish(input) + (beta[c] - gamma[c] * running_mean[c] / sqrt(running_var[c] + eps))
// Precompute: scale[c] = gamma[c] / sqrt(running_var[c] + eps)
//             bias[c] = beta[c] - gamma[c] * running_mean[c] / sqrt(running_var[c] + eps)
__global__ __launch_bounds__(256) void mish_bn_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ scale,  // precomputed scale per channel
    const float* __restrict__ bias,   // precomputed bias per channel
    float* __restrict__ output,
    int batch_size,
    int channels,
    int height,
    int width)
{
    int total_size = batch_size * channels * height * width;
    int spatial_size = height * width;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < total_size; i += stride) {
        // Compute channel index
        int c = (i / spatial_size) % channels;
        
        float x = input[i];
        float mish_x = fast_mish(x);
        output[i] = scale[c] * mish_x + bias[c];
    }
}

// Vectorized version for better memory throughput
__global__ __launch_bounds__(256) void mish_bn_fused_kernel_vec4(
    const float* __restrict__ input,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int height,
    int width)
{
    int total_size = batch_size * channels * height * width;
    int spatial_size = height * width;
    int channel_size = spatial_size;
    
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < total_size) {
        float4 in = *reinterpret_cast<const float4*>(input + idx);
        float4 out;
        
        // All 4 elements are adjacent in spatial dimension, so same channel
        int c = (idx / spatial_size) % channels;
        float s = scale[c];
        float b = bias[c];
        
        out.x = s * fast_mish(in.x) + b;
        out.y = s * fast_mish(in.y) + b;
        out.z = s * fast_mish(in.z) + b;
        out.w = s * fast_mish(in.w) + b;
        
        *reinterpret_cast<float4*>(output + idx) = out;
    } else if (idx < total_size) {
        for (int i = idx; i < total_size && i < idx + 4; i++) {
            int c = (i / spatial_size) % channels;
            float x = input[i];
            output[i] = scale[c] * fast_mish(x) + bias[c];
        }
    }
}

std::vector<torch::Tensor> mish_bn_fused_hip(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps)
{
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    auto output = torch::empty_like(input);
    
    // Precompute scale and bias for fused operation
    auto scale = gamma / torch::sqrt(running_var + eps);
    auto bias = beta - gamma * running_mean / torch::sqrt(running_var + eps);
    
    int total_size = batch_size * channels * height * width;
    const int block_size = 256;
    
    // Use vectorized version
    int num_vec4 = (total_size + 3) / 4;
    int num_blocks = (num_vec4 + block_size - 1) / block_size;
    
    mish_bn_fused_kernel_vec4<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        scale.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width
    );
    
    return {output};
}

// Simple mish kernel for training (where we can't fuse with BN due to mean/var computation)
__global__ __launch_bounds__(256) void mish_kernel_vec4(
    const float* __restrict__ input, 
    float* __restrict__ output, 
    int size) 
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < size) {
        float4 in = *reinterpret_cast<const float4*>(input + idx);
        float4 out;
        out.x = fast_mish(in.x);
        out.y = fast_mish(in.y);
        out.z = fast_mish(in.z);
        out.w = fast_mish(in.w);
        *reinterpret_cast<float4*>(output + idx) = out;
    } else if (idx < size) {
        for (int i = idx; i < size && i < idx + 4; i++) {
            output[i] = fast_mish(input[i]);
        }
    }
}

torch::Tensor mish_hip(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    int num_elements = (size + 3) / 4;
    int num_blocks = (num_elements + block_size - 1) / block_size;
    
    mish_kernel_vec4<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    return output;
}
"""

cpp_source = """
std::vector<torch::Tensor> mish_bn_fused_hip(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps);
torch::Tensor mish_hip(torch::Tensor input);
"""

fused_module = load_inline(
    name="fused_module",
    cpp_sources=cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["mish_bn_fused_hip", "mish_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "--offload-arch=gfx942", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model with fused Mish+BatchNorm kernel for inference.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.eps = eps

    def forward(self, x):
        x = self.conv(x)
        
        if self.training:
            # During training, we need to compute batch stats, so use separate ops
            x = fused_module.mish_hip(x)
            x = self.bn(x)
        else:
            # During inference, fuse Mish + BatchNorm
            result = fused_module.mish_bn_fused_hip(
                x,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.weight,
                self.bn.bias,
                self.eps
            )
            x = result[0]
        
        return x
