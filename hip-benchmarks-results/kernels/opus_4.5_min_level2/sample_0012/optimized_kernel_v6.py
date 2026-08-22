import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused divide + GELU kernel only (let nn.Linear handle matmul+bias)
fused_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <cmath>

__device__ __forceinline__ float gelu_exact(float x) {
    const float kSqrt2Inv = 0.7071067811865475f;
    return x * 0.5f * (1.0f + erff(x * kSqrt2Inv));
}

// Simple grid-stride kernel for fused divide + GELU
__global__ void fused_div_gelu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float divisor_inv,
    const int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process 4 elements per thread
    for (int i = idx * 4; i < total_elements; i += stride * 4) {
        if (i + 3 < total_elements) {
            float4 in_val = *reinterpret_cast<const float4*>(input + i);
            
            float4 out_val;
            out_val.x = gelu_exact(in_val.x * divisor_inv);
            out_val.y = gelu_exact(in_val.y * divisor_inv);
            out_val.z = gelu_exact(in_val.z * divisor_inv);
            out_val.w = gelu_exact(in_val.w * divisor_inv);
            
            *reinterpret_cast<float4*>(output + i) = out_val;
        } else {
            for (int j = 0; j < 4 && i + j < total_elements; j++) {
                output[i + j] = gelu_exact(input[i + j] * divisor_inv);
            }
        }
    }
}

// In-place version to save memory
__global__ void fused_div_gelu_inplace_kernel(
    float* __restrict__ data,
    const float divisor_inv,
    const int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx * 4; i < total_elements; i += stride * 4) {
        if (i + 3 < total_elements) {
            float4 val = *reinterpret_cast<const float4*>(data + i);
            
            float4 out_val;
            out_val.x = gelu_exact(val.x * divisor_inv);
            out_val.y = gelu_exact(val.y * divisor_inv);
            out_val.z = gelu_exact(val.z * divisor_inv);
            out_val.w = gelu_exact(val.w * divisor_inv);
            
            *reinterpret_cast<float4*>(data + i) = out_val;
        } else {
            for (int j = 0; j < 4 && i + j < total_elements; j++) {
                data[i + j] = gelu_exact(data[i + j] * divisor_inv);
            }
        }
    }
}

torch::Tensor fused_div_gelu_hip(torch::Tensor input, float divisor) {
    auto total = input.numel();
    auto output = torch::empty_like(input);
    
    float divisor_inv = 1.0f / divisor;
    
    // High occupancy configuration
    const int block_size = 256;
    const int num_blocks = std::min(2048, (int)((total + block_size * 4 - 1) / (block_size * 4)));
    
    fused_div_gelu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        divisor_inv,
        total
    );
    
    return output;
}

void fused_div_gelu_inplace_hip(torch::Tensor input, float divisor) {
    auto total = input.numel();
    float divisor_inv = 1.0f / divisor;
    
    const int block_size = 256;
    const int num_blocks = std::min(2048, (int)((total + block_size * 4 - 1) / (block_size * 4)));
    
    fused_div_gelu_inplace_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        divisor_inv,
        total
    );
}
"""

fused_cpp = """
torch::Tensor fused_div_gelu_hip(torch::Tensor input, float divisor);
void fused_div_gelu_inplace_hip(torch::Tensor input, float divisor);
"""

fused_module = load_inline(
    name="fused_module_86v6",
    cpp_sources=fused_cpp,
    cuda_sources=fused_source,
    functions=["fused_div_gelu_hip", "fused_div_gelu_inplace_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "--offload-arch=gfx942"]
)


class ModelNew(nn.Module):
    """
    Optimized model with fused divide + GELU kernel.
    Uses standard nn.Linear for matmul+bias (highly optimized rocBLAS).
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor
        self.fused_module = fused_module

    def forward(self, x):
        # Standard nn.Linear (matmul + bias) - highly optimized
        x = self.linear(x)
        # Fused divide + GELU
        x = self.fused_module.fused_div_gelu_hip(x, self.divisor)
        return x


batch_size = 1024
input_size = 8192
output_size = 8192
divisor = 10.0


def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]


def get_init_inputs():
    return [input_size, output_size, divisor]
