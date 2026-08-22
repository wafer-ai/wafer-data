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

__global__ void fused_scale_residual_kernel(const float* __restrict__ input,
                                             float* __restrict__ output,
                                             const float combined_factor,
                                             const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process 4 elements per thread for better memory throughput
    int idx4 = idx * 4;
    
    if (idx4 + 3 < size) {
        // Load 4 floats at once using float4
        float4 val = *reinterpret_cast<const float4*>(&input[idx4]);
        val.x *= combined_factor;
        val.y *= combined_factor;
        val.z *= combined_factor;
        val.w *= combined_factor;
        *reinterpret_cast<float4*>(&output[idx4]) = val;
    } else if (idx4 < size) {
        // Handle remaining elements
        for (int i = idx4; i < size && i < idx4 + 4; i++) {
            output[i] = input[i] * combined_factor;
        }
    }
}

torch::Tensor fused_scale_residual_hip(torch::Tensor input, float scaling_factor) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    // Combined factor: scaling_factor + 1.0 (for residual)
    float combined_factor = scaling_factor + 1.0f;
    
    const int block_size = 256;
    // Each thread processes 4 elements
    const int num_blocks = ((size + 3) / 4 + block_size - 1) / block_size;
    
    fused_scale_residual_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        combined_factor,
        size
    );
    
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
        # Perform linear transformation using PyTorch (uses optimized cuBLAS/rocBLAS)
        x = self.matmul(x)
        # Fused scaling and residual addition
        x = self.fused_op.fused_scale_residual_hip(x, self.scaling_factor)
        return x
