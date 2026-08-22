import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized Mish with larger work per thread and better ILP
mish_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

// Fast mish with numerical stability
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

// Process 16 elements per thread for better occupancy
__global__ __launch_bounds__(256) void mish_kernel_vec16(
    const float* __restrict__ input, 
    float* __restrict__ output, 
    int size) 
{
    int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    
    if (base_idx + 15 < size) {
        // Load 16 floats (4 float4s)
        float4 in0 = __builtin_nontemporal_load(reinterpret_cast<const float4*>(input + base_idx));
        float4 in1 = __builtin_nontemporal_load(reinterpret_cast<const float4*>(input + base_idx + 4));
        float4 in2 = __builtin_nontemporal_load(reinterpret_cast<const float4*>(input + base_idx + 8));
        float4 in3 = __builtin_nontemporal_load(reinterpret_cast<const float4*>(input + base_idx + 12));
        
        float4 out0, out1, out2, out3;
        
        // Process all 16 elements
        out0.x = fast_mish(in0.x);
        out0.y = fast_mish(in0.y);
        out0.z = fast_mish(in0.z);
        out0.w = fast_mish(in0.w);
        
        out1.x = fast_mish(in1.x);
        out1.y = fast_mish(in1.y);
        out1.z = fast_mish(in1.z);
        out1.w = fast_mish(in1.w);
        
        out2.x = fast_mish(in2.x);
        out2.y = fast_mish(in2.y);
        out2.z = fast_mish(in2.z);
        out2.w = fast_mish(in2.w);
        
        out3.x = fast_mish(in3.x);
        out3.y = fast_mish(in3.y);
        out3.z = fast_mish(in3.z);
        out3.w = fast_mish(in3.w);
        
        // Store 16 floats
        *reinterpret_cast<float4*>(output + base_idx) = out0;
        *reinterpret_cast<float4*>(output + base_idx + 4) = out1;
        *reinterpret_cast<float4*>(output + base_idx + 8) = out2;
        *reinterpret_cast<float4*>(output + base_idx + 12) = out3;
    } else {
        // Handle remaining elements
        for (int i = base_idx; i < size && i < base_idx + 16; i++) {
            output[i] = fast_mish(input[i]);
        }
    }
}

// Simpler version for cases where vec16 doesn't apply well
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
    
    // Use vec16 for large tensors divisible by 16
    if (size >= 16 && size % 16 == 0) {
        int num_elements = size / 16;
        int num_blocks = (num_elements + block_size - 1) / block_size;
        mish_kernel_vec16<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            size
        );
    } else {
        int num_elements = (size + 3) / 4;
        int num_blocks = (num_elements + block_size - 1) / block_size;
        mish_kernel_vec4<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            size
        );
    }
    
    return output;
}
"""

mish_cpp_source = """
torch::Tensor mish_hip(torch::Tensor input);
"""

mish_module = load_inline(
    name="mish_module",
    cpp_sources=mish_cpp_source,
    cuda_sources=mish_kernel_source,
    functions=["mish_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "--offload-arch=gfx942", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model with fused Mish activation kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.conv(x)
        # Fused Mish activation: x * tanh(softplus(x))
        x = mish_module.mish_hip(x)
        x = self.bn(x)
        return x
