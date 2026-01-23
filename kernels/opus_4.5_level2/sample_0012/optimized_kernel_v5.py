import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused divide + GELU kernel with 8-wide processing
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// GELU using erf (matches PyTorch default)
__device__ __forceinline__ float gelu_erf(float x) {
    return x * 0.5f * (1.0f + erff(x * 0.7071067811865475f));
}

// Process 8 elements per thread for better memory utilization on MI300X
__global__ void fused_div_gelu_kernel_wide(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float inv_divisor,
    const int size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Process 8 elements per iteration
    for (int base = tid * 8; base < size; base += stride * 8) {
        if (base + 7 < size) {
            // Load 8 elements as 2 float4
            float4 val0 = *reinterpret_cast<const float4*>(&input[base]);
            float4 val1 = *reinterpret_cast<const float4*>(&input[base + 4]);
            
            // Apply fused operation
            val0.x = gelu_erf(val0.x * inv_divisor);
            val0.y = gelu_erf(val0.y * inv_divisor);
            val0.z = gelu_erf(val0.z * inv_divisor);
            val0.w = gelu_erf(val0.w * inv_divisor);
            
            val1.x = gelu_erf(val1.x * inv_divisor);
            val1.y = gelu_erf(val1.y * inv_divisor);
            val1.z = gelu_erf(val1.z * inv_divisor);
            val1.w = gelu_erf(val1.w * inv_divisor);
            
            // Store results
            *reinterpret_cast<float4*>(&output[base]) = val0;
            *reinterpret_cast<float4*>(&output[base + 4]) = val1;
        } else {
            // Handle tail
            for (int i = base; i < size; i++) {
                output[i] = gelu_erf(input[i] * inv_divisor);
            }
        }
    }
}

torch::Tensor fused_div_gelu_hip(torch::Tensor input, float divisor) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const float inv_divisor = 1.0f / divisor;
    
    const int block_size = 256;
    const int elements_per_thread = 8;
    const int threads_needed = (size + elements_per_thread - 1) / elements_per_thread;
    const int num_blocks = std::min((threads_needed + block_size - 1) / block_size, 8192);
    
    fused_div_gelu_kernel_wide<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        inv_divisor,
        size
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_div_gelu_hip(torch::Tensor input, float divisor);
"""

fused_module = load_inline(
    name="fused_div_gelu_wide",
    cpp_sources=cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_div_gelu_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model using addmm and fused divide + GELU.
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.empty(output_size))
        self.divisor = divisor
        
        # Standard initialization
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Use F.linear which internally uses optimized rocBLAS
        x = F.linear(x, self.weight, self.bias)
        # Apply fused divide + GELU
        x = fused_module.fused_div_gelu_hip(x, self.divisor)
        return x
