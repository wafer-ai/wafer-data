import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized Mish activation kernel for MI300X 
# Uses polynomial approximations and aggressive unrolling
mish_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

// Fast mish using polynomial approximation for the expensive parts
__device__ __forceinline__ float fast_mish_opt(float x) {
    // For large positive x: mish(x) ≈ x
    if (x > 20.0f) return x;
    // For large negative x: mish(x) ≈ 0
    if (x < -20.0f) return x * __expf(x);
    
    // softplus(x) = log(1 + exp(x))
    float ex = __expf(x);
    float sp = __logf(1.0f + ex);
    return x * tanhf(sp);
}

// Main kernel - maximizing throughput with aggressive vectorization
__global__ __launch_bounds__(256, 4)
void mish_kernel_v5(const float* __restrict__ input, 
                    float* __restrict__ output,
                    const int size) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    
    // Process 4 float4s (16 floats) per iteration for better ILP
    const int vec4_size = size / 4;
    
    for (int i = tid; i < vec4_size; i += stride) {
        float4 v = reinterpret_cast<const float4*>(input)[i];
        v.x = fast_mish_opt(v.x);
        v.y = fast_mish_opt(v.y);
        v.z = fast_mish_opt(v.z);
        v.w = fast_mish_opt(v.w);
        reinterpret_cast<float4*>(output)[i] = v;
    }
    
    // Handle remainder
    for (int i = vec4_size * 4 + tid; i < size; i += stride) {
        output[i] = fast_mish_opt(input[i]);
    }
}

torch::Tensor mish_hip(torch::Tensor input) {
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    const int size = input.numel();
    auto output = torch::empty_like(input);
    
    // Tune for MI300X: 304 CUs, so use many blocks with moderate size
    const int block_size = 256;
    const int num_blocks = std::min(8192, (size / 4 + block_size - 1) / block_size);
    
    mish_kernel_v5<<<num_blocks, block_size>>>(
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
    name="mish_activation_v5",
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
