import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized fused kernel with shared memory and better occupancy
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Exact GELU using erf
__device__ __forceinline__ float gelu(float x) {
    return x * 0.5f * (1.0f + erff(x * 0.7071067811865475f));
}

// Kernel that processes entire rows, better for cache locality with output of matmul
__global__ __launch_bounds__(1024) void fused_div_gelu_rowwise(
    float* __restrict__ data,  // In-place operation
    const float inv_divisor,
    const int rows,
    const int cols
) {
    // Each block handles one row
    const int row = blockIdx.x;
    if (row >= rows) return;
    
    float* row_data = data + row * cols;
    
    // Process columns using all threads in the block
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        float val = row_data[col] * inv_divisor;
        row_data[col] = gelu(val);
    }
}

// Optimized kernel with vectorized loads/stores
__global__ void fused_div_gelu_vec(
    float* __restrict__ data,
    const float inv_divisor,
    const int size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = tid * 4;
    
    if (idx + 3 < size) {
        float4 val = *reinterpret_cast<float4*>(&data[idx]);
        
        val.x = gelu(val.x * inv_divisor);
        val.y = gelu(val.y * inv_divisor);
        val.z = gelu(val.z * inv_divisor);
        val.w = gelu(val.w * inv_divisor);
        
        *reinterpret_cast<float4*>(&data[idx]) = val;
    } else if (idx < size) {
        for (int i = idx; i < size; i++) {
            data[i] = gelu(data[i] * inv_divisor);
        }
    }
}

torch::Tensor fused_linear_div_gelu_hip(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    
    // Perform linear operation: output = input @ weight.T + bias
    auto output = torch::addmm(bias, input, weight.t());
    
    const int size = output.numel();
    const float inv_divisor = 1.0f / divisor;
    
    // Apply fused divide + GELU in-place
    const int block_size = 256;
    const int num_blocks = (size / 4 + block_size - 1) / block_size;
    
    fused_div_gelu_vec<<<num_blocks, block_size>>>(
        output.data_ptr<float>(),
        inv_divisor,
        size
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_linear_div_gelu_hip(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor
);
"""

fused_module = load_inline(
    name="fused_linear_div_gelu",
    cpp_sources=cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_linear_div_gelu_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Model with fused linear + div + gelu in a single call.
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.empty(output_size))
        self.divisor = divisor
        
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return fused_module.fused_linear_div_gelu_hip(x, self.weight, self.bias, self.divisor)
