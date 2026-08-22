import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused Mish + BatchNorm kernel for inference with better optimizations
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

// Vectorized fused Mish + BN with 16 elements per thread for maximum throughput
__global__ __launch_bounds__(256) void mish_bn_fused_kernel_vec16(
    const float* __restrict__ input,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int spatial_size)
{
    int total_size = batch_size * channels * spatial_size;
    int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    
    if (base_idx + 15 < total_size) {
        // Load 16 floats (4 float4s)
        float4 in0 = *reinterpret_cast<const float4*>(input + base_idx);
        float4 in1 = *reinterpret_cast<const float4*>(input + base_idx + 4);
        float4 in2 = *reinterpret_cast<const float4*>(input + base_idx + 8);
        float4 in3 = *reinterpret_cast<const float4*>(input + base_idx + 12);
        
        // Get channel for first element - all 16 elements likely in same channel for large spatial dims
        int c0 = (base_idx / spatial_size) % channels;
        float s0 = scale[c0];
        float b0 = bias[c0];
        
        float4 out0, out1, out2, out3;
        
        // Check if all 16 elements are in the same channel (common case for large spatial dims)
        int c_last = ((base_idx + 15) / spatial_size) % channels;
        
        if (c0 == c_last) {
            // Fast path: all same channel
            out0.x = s0 * fast_mish(in0.x) + b0;
            out0.y = s0 * fast_mish(in0.y) + b0;
            out0.z = s0 * fast_mish(in0.z) + b0;
            out0.w = s0 * fast_mish(in0.w) + b0;
            
            out1.x = s0 * fast_mish(in1.x) + b0;
            out1.y = s0 * fast_mish(in1.y) + b0;
            out1.z = s0 * fast_mish(in1.z) + b0;
            out1.w = s0 * fast_mish(in1.w) + b0;
            
            out2.x = s0 * fast_mish(in2.x) + b0;
            out2.y = s0 * fast_mish(in2.y) + b0;
            out2.z = s0 * fast_mish(in2.z) + b0;
            out2.w = s0 * fast_mish(in2.w) + b0;
            
            out3.x = s0 * fast_mish(in3.x) + b0;
            out3.y = s0 * fast_mish(in3.y) + b0;
            out3.z = s0 * fast_mish(in3.z) + b0;
            out3.w = s0 * fast_mish(in3.w) + b0;
        } else {
            // Slow path: handle channel boundaries
            #define PROCESS_ELEMENT(idx_offset, vec_in, component, vec_out) { \
                int idx = base_idx + idx_offset; \
                int c = (idx / spatial_size) % channels; \
                float s = scale[c]; \
                float b = bias[c]; \
                vec_out.component = s * fast_mish(vec_in.component) + b; \
            }
            
            PROCESS_ELEMENT(0, in0, x, out0);
            PROCESS_ELEMENT(1, in0, y, out0);
            PROCESS_ELEMENT(2, in0, z, out0);
            PROCESS_ELEMENT(3, in0, w, out0);
            PROCESS_ELEMENT(4, in1, x, out1);
            PROCESS_ELEMENT(5, in1, y, out1);
            PROCESS_ELEMENT(6, in1, z, out1);
            PROCESS_ELEMENT(7, in1, w, out1);
            PROCESS_ELEMENT(8, in2, x, out2);
            PROCESS_ELEMENT(9, in2, y, out2);
            PROCESS_ELEMENT(10, in2, z, out2);
            PROCESS_ELEMENT(11, in2, w, out2);
            PROCESS_ELEMENT(12, in3, x, out3);
            PROCESS_ELEMENT(13, in3, y, out3);
            PROCESS_ELEMENT(14, in3, z, out3);
            PROCESS_ELEMENT(15, in3, w, out3);
            
            #undef PROCESS_ELEMENT
        }
        
        // Store 16 floats
        *reinterpret_cast<float4*>(output + base_idx) = out0;
        *reinterpret_cast<float4*>(output + base_idx + 4) = out1;
        *reinterpret_cast<float4*>(output + base_idx + 8) = out2;
        *reinterpret_cast<float4*>(output + base_idx + 12) = out3;
    } else if (base_idx < total_size) {
        // Handle remaining elements
        for (int i = base_idx; i < total_size && i < base_idx + 16; i++) {
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
    auto spatial_size = height * width;
    
    auto output = torch::empty_like(input);
    
    // Precompute scale and bias for fused operation
    auto inv_std = torch::rsqrt(running_var + eps);
    auto scale = gamma * inv_std;
    auto bias = beta - gamma * running_mean * inv_std;
    
    int total_size = batch_size * channels * spatial_size;
    const int block_size = 256;
    
    // Use vec16 kernel
    int num_vec16 = (total_size + 15) / 16;
    int num_blocks = (num_vec16 + block_size - 1) / block_size;
    
    mish_bn_fused_kernel_vec16<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        scale.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        spatial_size
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
