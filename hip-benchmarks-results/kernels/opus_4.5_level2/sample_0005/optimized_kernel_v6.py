import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused Mish activation with maximum bandwidth utilization
mish_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

// Compute mish: x * tanh(softplus(x))
__device__ __forceinline__ float fast_mish(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return x * __expf(x);
    float sp = __logf(1.0f + __expf(x));
    return x * tanhf(sp);
}

// Kernel optimized for MI300X bandwidth
// MI300X has very high memory bandwidth - we need to maximize coalescing
__global__ void mish_kernel_bandwidth(const float* __restrict__ input, 
                                       float* __restrict__ output,
                                       const int n) {
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // Each thread handles a float4 (128 bits = 4 floats)
    int vec_n = n >> 2;  // n / 4
    
    for (int i = idx; i < vec_n; i += stride) {
        float4 val = __builtin_nontemporal_load(reinterpret_cast<const float4*>(input) + i);
        
        float4 result;
        result.x = fast_mish(val.x);
        result.y = fast_mish(val.y);
        result.z = fast_mish(val.z);
        result.w = fast_mish(val.w);
        
        __builtin_nontemporal_store(result, reinterpret_cast<float4*>(output) + i);
    }
    
    // Handle remaining elements
    int base = vec_n << 2;
    for (int i = base + idx; i < n; i += stride) {
        output[i] = fast_mish(input[i]);
    }
}

torch::Tensor mish_hip(torch::Tensor input) {
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    const int n = input.numel();
    auto output = torch::empty_like(input);
    
    // Use 512 threads per block for good occupancy
    const int block_size = 512;
    // Many blocks to saturate all CUs
    const int num_blocks = std::min(16384, (n / 4 + block_size - 1) / block_size);
    
    mish_kernel_bandwidth<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        n
    );
    
    return output;
}
"""

mish_cpp_source = """
torch::Tensor mish_hip(torch::Tensor input);
"""

mish_module = load_inline(
    name="mish_activation_v6",
    cpp_sources=mish_cpp_source,
    cuda_sources=mish_kernel_source,
    functions=["mish_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math", "-munsafe-fp-atomics"]
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
