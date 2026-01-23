import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized Mish activation kernel 
# Uses larger work per thread, better occupancy settings
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

// Kernel optimized for MI300X - 8 elements per thread, grid-stride loop
__global__ __launch_bounds__(256, 8)
void mish_kernel_fast(const float* __restrict__ input, 
                      float* __restrict__ output,
                      const int size) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    
    // Process 8 floats per iteration using two float4s
    const int vec_size = size / 8;
    
    for (int i = tid; i < vec_size; i += total_threads) {
        const int base_idx = i * 8;
        
        // Load 8 floats as two float4s
        float4 in0 = reinterpret_cast<const float4*>(input + base_idx)[0];
        float4 in1 = reinterpret_cast<const float4*>(input + base_idx)[1];
        
        float4 out0, out1;
        out0.x = fast_mish(in0.x);
        out0.y = fast_mish(in0.y);
        out0.z = fast_mish(in0.z);
        out0.w = fast_mish(in0.w);
        out1.x = fast_mish(in1.x);
        out1.y = fast_mish(in1.y);
        out1.z = fast_mish(in1.z);
        out1.w = fast_mish(in1.w);
        
        reinterpret_cast<float4*>(output + base_idx)[0] = out0;
        reinterpret_cast<float4*>(output + base_idx)[1] = out1;
    }
    
    // Handle remainder (last elements that don't fit in 8-element chunks)
    const int remainder_start = vec_size * 8;
    for (int i = remainder_start + tid; i < size; i += total_threads) {
        output[i] = fast_mish(input[i]);
    }
}

torch::Tensor mish_hip(torch::Tensor input) {
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    const auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    // Launch enough blocks to saturate MI300X (many CUs)
    const int num_blocks = std::min(2048, (int)((size / 8 + block_size - 1) / block_size));
    
    mish_kernel_fast<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        size
    );
    
    return output;
}
"""

mish_cpp_source = """
torch::Tensor mish_hip(torch::Tensor input);
"""

mish_module = load_inline(
    name="mish_activation_v3",
    cpp_sources=mish_cpp_source,
    cuda_sources=mish_kernel_source,
    functions=["mish_hip"],
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
        x = self.mish.mish_hip(x)
        x = self.bn(x)
        return x


def get_inputs():
    return [torch.rand(64, 64, 128, 128).cuda()]


def get_init_inputs():
    return [64, 128, 3]
