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

// Kernel for fusing bias add, division, and GELU
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
    // Each thread handles one element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * output_size;
    
    // Process 4 elements per thread
    int idx4 = idx * 4;
    
    if (idx4 + 3 < total) {
        // Calculate positions
        int col0 = (idx4) % output_size;
        int col1 = (idx4 + 1) % output_size;
        int col2 = (idx4 + 2) % output_size;
        int col3 = (idx4 + 3) % output_size;
        
        // Load input values
        float4 in_val = *reinterpret_cast<const float4*>(input + idx4);
        
        // Load bias values
        float b0 = bias[col0];
        float b1 = bias[col1];
        float b2 = bias[col2];
        float b3 = bias[col3];
        
        // Apply bias, division, and GELU
        float x0 = (in_val.x + b0) * divisor_inv;
        float x1 = (in_val.y + b1) * divisor_inv;
        float x2 = (in_val.z + b2) * divisor_inv;
        float x3 = (in_val.w + b3) * divisor_inv;
        
        float4 out_val;
        out_val.x = gelu_exact(x0);
        out_val.y = gelu_exact(x1);
        out_val.z = gelu_exact(x2);
        out_val.w = gelu_exact(x3);
        
        *reinterpret_cast<float4*>(output + idx4) = out_val;
    } else {
        // Handle remaining elements
        for (int i = 0; i < 4 && idx4 + i < total; i++) {
            int pos = idx4 + i;
            int col = pos % output_size;
            float x = (input[pos] + bias[col]) * divisor_inv;
            output[pos] = gelu_exact(x);
        }
    }
}

// Alternative: row-wise processing for better cache efficiency
__global__ void fused_bias_div_gelu_kernel_v2(
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
    
    fused_bias_div_gelu_kernel_v2<<<num_blocks, block_size>>>(
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
    name="fused_ops",
    cpp_sources=fused_bias_div_gelu_cpp,
    cuda_sources=fused_bias_div_gelu_source,
    functions=["fused_bias_div_gelu_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "--offload-arch=gfx942"]
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses bias add, divide, and GELU into a single kernel.
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        # Just use the weight matrix, bias handled separately
        self.weight = nn.Parameter(torch.randn(output_size, input_size) * 0.02)
        self.bias = nn.Parameter(torch.zeros(output_size))
        self.divisor = divisor
        
        # Copy initialization from nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
        self.fused_ops = fused_ops

    def forward(self, x):
        # Matmul without bias (use F.linear with bias=None or mm)
        x = F.linear(x, self.weight, bias=None)
        # Fused bias + divide + GELU
        x = self.fused_ops.fused_bias_div_gelu_hip(x, self.bias, self.divisor)
        return x


batch_size = 1024
input_size = 8192
output_size = 8192
divisor = 10.0


def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]


def get_init_inputs():
    return [input_size, output_size, divisor]
