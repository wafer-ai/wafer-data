import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused bias + divide + GELU kernel
fused_bias_div_gelu_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <cmath>

__device__ __forceinline__ float gelu_exact(float x) {
    // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    const float kSqrt2Inv = 0.7071067811865475f;
    return x * 0.5f * (1.0f + erff(x * kSqrt2Inv));
}

// Row-wise processing for better cache efficiency
// Input shape: (batch_size, output_size)
// Bias shape: (output_size,)
__global__ void fused_bias_div_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const float divisor_inv,
    const int batch_size,
    const int output_size)
{
    // Block handles multiple rows, threads handle columns
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row >= batch_size) return;
    
    const float* row_in = input + row * output_size;
    float* row_out = output + row * output_size;
    
    // Each thread processes multiple elements with stride
    for (int col = tid * 4; col < output_size; col += blockDim.x * 4) {
        if (col + 3 < output_size) {
            float4 in_val = *reinterpret_cast<const float4*>(row_in + col);
            float4 b_val = *reinterpret_cast<const float4*>(bias + col);
            
            float x0 = (in_val.x + b_val.x) * divisor_inv;
            float x1 = (in_val.y + b_val.y) * divisor_inv;
            float x2 = (in_val.z + b_val.z) * divisor_inv;
            float x3 = (in_val.w + b_val.w) * divisor_inv;
            
            float4 out_val;
            out_val.x = gelu_exact(x0);
            out_val.y = gelu_exact(x1);
            out_val.z = gelu_exact(x2);
            out_val.w = gelu_exact(x3);
            
            *reinterpret_cast<float4*>(row_out + col) = out_val;
        } else {
            for (int i = 0; i < 4 && col + i < output_size; i++) {
                float x = (row_in[col + i] + bias[col + i]) * divisor_inv;
                row_out[col + i] = gelu_exact(x);
            }
        }
    }
}

torch::Tensor fused_bias_div_gelu_hip(torch::Tensor input, torch::Tensor bias, float divisor) {
    auto batch_size = input.size(0);
    auto output_size = input.size(1);
    auto output = torch::empty_like(input);
    
    float divisor_inv = 1.0f / divisor;
    
    // Use row-wise kernel
    int block_size = 256;
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

fused_bias_div_gelu_cpp = """
torch::Tensor fused_bias_div_gelu_hip(torch::Tensor input, torch::Tensor bias, float divisor);
"""

fused_ops = load_inline(
    name="fused_ops_v3",
    cpp_sources=fused_bias_div_gelu_cpp,
    cuda_sources=fused_bias_div_gelu_source,
    functions=["fused_bias_div_gelu_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "--offload-arch=gfx942"]
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses bias add, divide, and GELU into a single kernel.
    Uses nn.Linear's weight and bias, but applies bias separately in fused kernel.
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        # Use nn.Linear to get proper initialization
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor
        self.fused_ops = fused_ops

    def forward(self, x):
        # Matmul without bias
        x = F.linear(x, self.linear.weight, bias=None)
        # Fused bias + divide + GELU
        x = self.fused_ops.fused_bias_div_gelu_hip(x, self.linear.bias, self.divisor)
        return x


batch_size = 1024
input_size = 8192
output_size = 8192
divisor = 10.0


def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]


def get_init_inputs():
    return [input_size, output_size, divisor]
