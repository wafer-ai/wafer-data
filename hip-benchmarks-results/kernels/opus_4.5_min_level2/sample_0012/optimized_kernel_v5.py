import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused bias + divide + GELU kernel
fused_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <cmath>

__device__ __forceinline__ float gelu_exact(float x) {
    // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    const float kSqrt2Inv = 0.7071067811865475f;
    return x * 0.5f * (1.0f + erff(x * kSqrt2Inv));
}

// Optimized kernel with higher thread count and vectorization
__global__ void fused_bias_div_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const float divisor_inv,
    const int batch_size,
    const int output_size)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row >= batch_size) return;
    
    const float* row_in = input + row * output_size;
    float* row_out = output + row * output_size;
    
    // Process 8 elements per iteration per thread
    for (int col = tid * 8; col < output_size; col += blockDim.x * 8) {
        if (col + 7 < output_size) {
            // Load 2 float4s
            float4 in_val0 = *reinterpret_cast<const float4*>(row_in + col);
            float4 in_val1 = *reinterpret_cast<const float4*>(row_in + col + 4);
            float4 b_val0 = *reinterpret_cast<const float4*>(bias + col);
            float4 b_val1 = *reinterpret_cast<const float4*>(bias + col + 4);
            
            float4 out_val0, out_val1;
            out_val0.x = gelu_exact((in_val0.x + b_val0.x) * divisor_inv);
            out_val0.y = gelu_exact((in_val0.y + b_val0.y) * divisor_inv);
            out_val0.z = gelu_exact((in_val0.z + b_val0.z) * divisor_inv);
            out_val0.w = gelu_exact((in_val0.w + b_val0.w) * divisor_inv);
            
            out_val1.x = gelu_exact((in_val1.x + b_val1.x) * divisor_inv);
            out_val1.y = gelu_exact((in_val1.y + b_val1.y) * divisor_inv);
            out_val1.z = gelu_exact((in_val1.z + b_val1.z) * divisor_inv);
            out_val1.w = gelu_exact((in_val1.w + b_val1.w) * divisor_inv);
            
            *reinterpret_cast<float4*>(row_out + col) = out_val0;
            *reinterpret_cast<float4*>(row_out + col + 4) = out_val1;
        } else {
            // Handle edge case
            for (int i = 0; i < 8 && col + i < output_size; i++) {
                float x = (row_in[col + i] + bias[col + i]) * divisor_inv;
                row_out[col + i] = gelu_exact(x);
            }
        }
    }
}

torch::Tensor fused_matmul_bias_div_gelu_hip(torch::Tensor input, torch::Tensor bias, float divisor) {
    auto batch_size = input.size(0);
    auto output_size = input.size(1);
    auto output = torch::empty_like(input);
    
    float divisor_inv = 1.0f / divisor;
    
    int block_size = 1024;
    int num_blocks = batch_size;
    
    fused_bias_div_gelu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        divisor_inv,
        batch_size,
        output_size
    );
    
    return output;
}
"""

fused_cpp = """
torch::Tensor fused_matmul_bias_div_gelu_hip(torch::Tensor input, torch::Tensor bias, float divisor);
"""

fused_module = load_inline(
    name="fused_module_86v5",
    cpp_sources=fused_cpp,
    cuda_sources=fused_source,
    functions=["fused_matmul_bias_div_gelu_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "--offload-arch=gfx942"]
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses bias add, divide, and GELU into a single kernel.
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor
        self.fused_module = fused_module

    def forward(self, x):
        # Matmul without bias (use rocBLAS via F.linear)
        x = F.linear(x, self.linear.weight, bias=None)
        # Fused bias + divide + GELU
        x = self.fused_module.fused_matmul_bias_div_gelu_hip(x, self.linear.bias, self.divisor)
        return x


batch_size = 1024
input_size = 8192
output_size = 8192
divisor = 10.0


def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]


def get_init_inputs():
    return [input_size, output_size, divisor]
