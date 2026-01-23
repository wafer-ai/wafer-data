import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized fused kernel for in-place scaling + residual
# Computation: output = input * (1 + scaling_factor)
fused_scale_residual_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Optimized kernel using float4 vectorization
// Process 4 float elements per thread
__global__ void fused_scale_residual_inplace_kernel(float* __restrict__ data,
                                                     const float combined_factor,
                                                     const int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < size) {
        // Load 4 floats (128 bits) at once
        float4* ptr = reinterpret_cast<float4*>(&data[idx]);
        float4 val = *ptr;
        
        // Apply combined scaling factor
        val.x *= combined_factor;
        val.y *= combined_factor;
        val.z *= combined_factor;
        val.w *= combined_factor;
        
        // Store back
        *ptr = val;
    } else {
        // Handle remainder
        for (int i = idx; i < size && i < idx + 4; i++) {
            data[i] *= combined_factor;
        }
    }
}

torch::Tensor fused_scale_residual_inplace_hip(torch::Tensor input, float scaling_factor) {
    auto size = input.numel();
    
    // Combined factor: scaling_factor + 1.0 (for residual)
    float combined_factor = scaling_factor + 1.0f;
    
    const int block_size = 256;
    const int elements_per_thread = 4;
    const int num_blocks = (size + (block_size * elements_per_thread) - 1) / (block_size * elements_per_thread);
    
    fused_scale_residual_inplace_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        combined_factor,
        size
    );
    
    return input;
}
"""

fused_cpp_source = """
torch::Tensor fused_scale_residual_inplace_hip(torch::Tensor input, float scaling_factor);
"""

fused_scale_residual = load_inline(
    name="fused_scale_residual",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_scale_residual_source,
    functions=["fused_scale_residual_inplace_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"],
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses scaling and residual addition.
    
    The original computation: x * scaling_factor + x
    Optimized to: x * (1 + scaling_factor) done in-place
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.fused_op = fused_scale_residual

    def forward(self, x):
        # Perform linear transformation using PyTorch (uses optimized rocBLAS)
        x = self.matmul(x)
        # Fused scaling and residual addition (in-place)
        x = self.fused_op.fused_scale_residual_inplace_hip(x, self.scaling_factor)
        return x
