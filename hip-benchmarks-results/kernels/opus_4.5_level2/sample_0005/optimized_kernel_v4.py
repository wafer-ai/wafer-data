import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized Mish activation using in-place operation to reduce memory footprint
mish_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

// Fast mish using device intrinsics
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

// Kernel with maximum work per thread using float4 and grid-stride
__global__ __launch_bounds__(1024, 2)
void mish_kernel_inplace(float* __restrict__ data,
                         const int size) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    const int vec_size = size / 4;
    
    // Process 4 floats at a time
    for (int i = tid; i < vec_size; i += total_threads) {
        float4 val = reinterpret_cast<float4*>(data)[i];
        val.x = fast_mish(val.x);
        val.y = fast_mish(val.y);
        val.z = fast_mish(val.z);
        val.w = fast_mish(val.w);
        reinterpret_cast<float4*>(data)[i] = val;
    }
    
    // Handle remainder
    const int remainder_start = vec_size * 4;
    for (int i = remainder_start + tid; i < size; i += total_threads) {
        data[i] = fast_mish(data[i]);
    }
}

// Non-inplace version that processes output of conv directly
__global__ __launch_bounds__(1024, 2)
void mish_kernel_fast(const float* __restrict__ input, 
                      float* __restrict__ output,
                      const int size) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    const int vec_size = size / 4;
    
    // Process 4 floats at a time
    for (int i = tid; i < vec_size; i += total_threads) {
        float4 in_val = reinterpret_cast<const float4*>(input)[i];
        float4 out_val;
        out_val.x = fast_mish(in_val.x);
        out_val.y = fast_mish(in_val.y);
        out_val.z = fast_mish(in_val.z);
        out_val.w = fast_mish(in_val.w);
        reinterpret_cast<float4*>(output)[i] = out_val;
    }
    
    // Handle remainder
    const int remainder_start = vec_size * 4;
    for (int i = remainder_start + tid; i < size; i += total_threads) {
        output[i] = fast_mish(input[i]);
    }
}

torch::Tensor mish_hip(torch::Tensor input) {
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    const int size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 1024;
    // MI300X has 304 CUs, we want good occupancy
    const int num_blocks = std::min(4096, (size / 4 + block_size - 1) / block_size);
    
    mish_kernel_fast<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        size
    );
    
    return output;
}

// Inplace version
torch::Tensor mish_hip_inplace(torch::Tensor input) {
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    const int size = input.numel();
    
    const int block_size = 1024;
    const int num_blocks = std::min(4096, (size / 4 + block_size - 1) / block_size);
    
    mish_kernel_inplace<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        size
    );
    
    return input;
}
"""

mish_cpp_source = """
torch::Tensor mish_hip(torch::Tensor input);
torch::Tensor mish_hip_inplace(torch::Tensor input);
"""

mish_module = load_inline(
    name="mish_activation_v4",
    cpp_sources=mish_cpp_source,
    cuda_sources=mish_kernel_source,
    functions=["mish_hip", "mish_hip_inplace"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model with fused Mish activation kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.mish = mish_module

    def forward(self, x):
        x = self.conv(x)
        # Use in-place mish to save memory bandwidth
        x = self.mish.mish_hip_inplace(x.contiguous())
        x = self.bn(x)
        return x


def get_inputs():
    return [torch.rand(64, 64, 128, 128).cuda()]


def get_init_inputs():
    return [64, 128, 3]
