import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused kernel using vectorized loads
fused_scale_residual_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Highly optimized kernel for AMD MI300X
// Uses float4 vectorization for maximum memory bandwidth utilization
__global__ void fused_scale_residual_kernel_v3(const float* __restrict__ input,
                                                float* __restrict__ output,
                                                const float combined_factor,
                                                const int size) {
    // Calculate global index for float4 operations
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_float4 = size >> 2;  // size / 4
    
    // Main loop - process float4 vectors with grid-stride loop
    for (int idx = tid; idx < total_float4; idx += gridDim.x * blockDim.x) {
        const float4* in_ptr = reinterpret_cast<const float4*>(input);
        float4* out_ptr = reinterpret_cast<float4*>(output);
        
        float4 val = in_ptr[idx];
        val.x *= combined_factor;
        val.y *= combined_factor;
        val.z *= combined_factor;
        val.w *= combined_factor;
        out_ptr[idx] = val;
    }
    
    // Handle remainder (only first few threads)
    int remainder_start = total_float4 << 2;
    if (tid < (size - remainder_start)) {
        output[remainder_start + tid] = input[remainder_start + tid] * combined_factor;
    }
}

torch::Tensor fused_scale_residual_hip(torch::Tensor input, float scaling_factor) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    // Combined factor: scaling_factor + 1.0 (for residual)
    float combined_factor = scaling_factor + 1.0f;
    
    // Optimal block size for MI300X (wavefront size is 64)
    const int block_size = 256;
    // Use enough blocks to saturate all CUs (MI300X has 304 CUs)
    const int total_float4 = size >> 2;
    int num_blocks = (total_float4 + block_size - 1) / block_size;
    // Cap at 2048 blocks for grid-stride pattern
    num_blocks = (num_blocks > 2048) ? 2048 : num_blocks;
    
    fused_scale_residual_kernel_v3<<<num_blocks, block_size>>>(
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
        # Perform linear transformation using PyTorch (uses optimized rocBLAS)
        x = self.matmul(x)
        # Fused scaling and residual addition
        x = self.fused_op.fused_scale_residual_hip(x, self.scaling_factor)
        return x
