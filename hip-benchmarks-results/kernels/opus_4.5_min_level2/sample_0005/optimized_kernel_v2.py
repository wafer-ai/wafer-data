import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized Mish activation kernel with better memory access patterns
mish_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

// Fast approximation of mish for better performance
__device__ __forceinline__ float fast_mish(float x) {
    // softplus(x) = ln(1 + exp(x))
    float sp;
    if (x > 20.0f) {
        sp = x;
    } else if (x < -20.0f) {
        sp = expf(x);
    } else {
        sp = log1pf(expf(x));
    }
    return x * tanhf(sp);
}

// Vectorized kernel using float4 for maximum memory throughput
__global__ void mish_kernel_vec4(const float* __restrict__ input, 
                                  float* __restrict__ output, 
                                  int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < size) {
        // Load 4 floats at once
        float4 in = *reinterpret_cast<const float4*>(input + idx);
        
        float4 out;
        out.x = fast_mish(in.x);
        out.y = fast_mish(in.y);
        out.z = fast_mish(in.z);
        out.w = fast_mish(in.w);
        
        // Store 4 floats at once
        *reinterpret_cast<float4*>(output + idx) = out;
    } else if (idx < size) {
        // Handle remaining elements
        for (int i = idx; i < size && i < idx + 4; i++) {
            output[i] = fast_mish(input[i]);
        }
    }
}

// Even more aggressive vectorization with float4x2 (8 elements per thread)
__global__ void mish_kernel_vec8(const float* __restrict__ input, 
                                  float* __restrict__ output, 
                                  int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    
    if (idx + 7 < size) {
        // Load 8 floats (2 float4s)
        float4 in0 = *reinterpret_cast<const float4*>(input + idx);
        float4 in1 = *reinterpret_cast<const float4*>(input + idx + 4);
        
        float4 out0, out1;
        out0.x = fast_mish(in0.x);
        out0.y = fast_mish(in0.y);
        out0.z = fast_mish(in0.z);
        out0.w = fast_mish(in0.w);
        out1.x = fast_mish(in1.x);
        out1.y = fast_mish(in1.y);
        out1.z = fast_mish(in1.z);
        out1.w = fast_mish(in1.w);
        
        // Store 8 floats
        *reinterpret_cast<float4*>(output + idx) = out0;
        *reinterpret_cast<float4*>(output + idx + 4) = out1;
    } else if (idx < size) {
        // Handle remaining elements
        for (int i = idx; i < size && i < idx + 8; i++) {
            output[i] = fast_mish(input[i]);
        }
    }
}

torch::Tensor mish_hip(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    
    // Use vec8 for large tensors
    if (size >= 8 && size % 8 == 0) {
        int num_elements = size / 8;
        int num_blocks = (num_elements + block_size - 1) / block_size;
        mish_kernel_vec8<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            size
        );
    } else if (size >= 4) {
        int num_elements = (size + 3) / 4;
        int num_blocks = (num_elements + block_size - 1) / block_size;
        mish_kernel_vec4<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            size
        );
    } else {
        // Fallback for very small tensors
        mish_kernel_vec4<<<1, block_size>>>(
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
