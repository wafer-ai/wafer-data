import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused divide + GELU kernel
fused_div_gelu_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__device__ __forceinline__ float gelu_approx(float x) {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void fused_div_gelu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float inv_divisor,
    const int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process 4 elements per thread for better memory throughput
    int idx4 = idx * 4;
    
    if (idx4 + 3 < size) {
        // Load 4 elements
        float4 in_vec = *reinterpret_cast<const float4*>(&input[idx4]);
        
        // Apply division and GELU
        float4 out_vec;
        out_vec.x = gelu_approx(in_vec.x * inv_divisor);
        out_vec.y = gelu_approx(in_vec.y * inv_divisor);
        out_vec.z = gelu_approx(in_vec.z * inv_divisor);
        out_vec.w = gelu_approx(in_vec.w * inv_divisor);
        
        // Store 4 elements
        *reinterpret_cast<float4*>(&output[idx4]) = out_vec;
    } else if (idx4 < size) {
        // Handle remaining elements
        for (int i = idx4; i < size && i < idx4 + 4; i++) {
            float val = input[i] * inv_divisor;
            output[i] = gelu_approx(val);
        }
    }
}

torch::Tensor fused_div_gelu_hip(torch::Tensor input, float divisor) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    
    auto output = torch::empty_like(input);
    int size = input.numel();
    
    const float inv_divisor = 1.0f / divisor;
    
    const int block_size = 256;
    // Each thread handles 4 elements
    const int num_blocks = (size + block_size * 4 - 1) / (block_size * 4);
    
    fused_div_gelu_kernel<<<num_blocks, block_size>>>(
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
    name="fused_div_gelu",
    cpp_sources=cpp_source,
    cuda_sources=fused_div_gelu_source,
    functions=["fused_div_gelu_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses divide and GELU operations.
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor

    def forward(self, x):
        # Use PyTorch's optimized linear (rocBLAS)
        x = self.linear(x)
        # Use fused divide + GELU kernel
        x = fused_module.fused_div_gelu_hip(x, self.divisor)
        return x
