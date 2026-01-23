import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# In-place fused divide + GELU kernel
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Fast GELU using erf (matches PyTorch's default GELU)
__device__ __forceinline__ float gelu_erf(float x) {
    return x * 0.5f * (1.0f + erff(x * 0.7071067811865475f));
}

// Ultra fast GELU approximation using tanh
__device__ __forceinline__ float gelu_tanh(float x) {
    const float c1 = 0.7978845608028654f;
    const float c2 = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(c1 * (x + c2 * x3)));
}

__global__ void fused_div_gelu_inplace_kernel(
    float* __restrict__ data,
    const float inv_divisor,
    const int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process 4 elements per thread
    for (int base = idx * 4; base < size; base += stride * 4) {
        if (base + 3 < size) {
            float4 val = *reinterpret_cast<float4*>(&data[base]);
            
            val.x = gelu_erf(val.x * inv_divisor);
            val.y = gelu_erf(val.y * inv_divisor);
            val.z = gelu_erf(val.z * inv_divisor);
            val.w = gelu_erf(val.w * inv_divisor);
            
            *reinterpret_cast<float4*>(&data[base]) = val;
        } else {
            for (int i = base; i < size && i < base + 4; i++) {
                data[i] = gelu_erf(data[i] * inv_divisor);
            }
        }
    }
}

void fused_div_gelu_inplace_hip(torch::Tensor input, float divisor) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    
    const int size = input.numel();
    const float inv_divisor = 1.0f / divisor;
    
    const int block_size = 256;
    const int elements_per_thread = 4;
    const int threads_needed = (size + elements_per_thread - 1) / elements_per_thread;
    const int num_blocks = std::min((threads_needed + block_size - 1) / block_size, 4096);
    
    fused_div_gelu_inplace_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        inv_divisor,
        size
    );
}
"""

cpp_source = """
void fused_div_gelu_inplace_hip(torch::Tensor input, float divisor);
"""

fused_module = load_inline(
    name="fused_div_gelu_inplace",
    cpp_sources=cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_div_gelu_inplace_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model with in-place fused divide + GELU.
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor

    def forward(self, x):
        # Use PyTorch's optimized linear layer
        x = self.linear(x)
        # Apply fused divide + GELU in-place
        fused_module.fused_div_gelu_inplace_hip(x, self.divisor)
        return x
