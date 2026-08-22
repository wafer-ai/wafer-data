import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel that computes: output = matmul_result * (1 + scaling_factor)
# This combines scaling and residual addition into one operation
fused_scale_residual_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Kernel optimized for MI300X with float4 vectorization
__global__ void fused_scale_residual_kernel(const float* __restrict__ input,
                                             float* __restrict__ output,
                                             const float combined_factor,
                                             const int total_float4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_float4) {
        const float4* in_ptr = reinterpret_cast<const float4*>(input);
        float4* out_ptr = reinterpret_cast<float4*>(output);
        
        float4 val = in_ptr[idx];
        val.x *= combined_factor;
        val.y *= combined_factor;
        val.z *= combined_factor;
        val.w *= combined_factor;
        out_ptr[idx] = val;
    }
}

// Kernel for handling remainder elements
__global__ void fused_scale_residual_remainder(const float* __restrict__ input,
                                                float* __restrict__ output,
                                                const float combined_factor,
                                                const int start,
                                                const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx < size) {
        output[idx] = input[idx] * combined_factor;
    }
}

torch::Tensor fused_scale_residual_hip(torch::Tensor input, float scaling_factor) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    // Combined factor: scaling_factor + 1.0 (for residual)
    float combined_factor = scaling_factor + 1.0f;
    
    const int block_size = 256;
    const int total_float4 = size / 4;
    const int num_blocks = (total_float4 + block_size - 1) / block_size;
    
    if (num_blocks > 0) {
        fused_scale_residual_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            combined_factor,
            total_float4
        );
    }
    
    // Handle remainder
    int remainder = size - (total_float4 * 4);
    if (remainder > 0) {
        fused_scale_residual_remainder<<<1, remainder>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            combined_factor,
            total_float4 * 4,
            size
        );
    }
    
    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_scale_residual_hip(torch::Tensor input, float scaling_factor);
"""

fused_scale_residual = load_inline(
    name="fused_scale_residual",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_scale_residual_source,
    functions=["fused_scale_residual_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"],
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses scaling and residual addition.
    
    The original computation: x * scaling_factor + x
    Optimized to: x * (1 + scaling_factor)
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.fused_op = fused_scale_residual

    def forward(self, x):
        # Perform linear transformation using PyTorch (uses optimized rocBLAS)
        x = self.matmul(x)
        # Fused scaling and residual addition
        x = self.fused_op.fused_scale_residual_hip(x, self.scaling_factor)
        return x
