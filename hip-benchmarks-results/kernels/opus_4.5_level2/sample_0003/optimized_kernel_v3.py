import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# In-place fused kernel for scaling + residual add
fused_scale_residual_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// In-place version: avoids memory allocation overhead
__global__ void fused_scale_residual_inplace_kernel(float4* __restrict__ data,
                                                     float combined_factor,
                                                     int num_float4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop for better occupancy
    for (int i = idx; i < num_float4; i += blockDim.x * gridDim.x) {
        float4 val = data[i];
        val.x *= combined_factor;
        val.y *= combined_factor;
        val.z *= combined_factor;
        val.w *= combined_factor;
        data[i] = val;
    }
}

// Handle remainder elements in-place
__global__ void fused_scale_residual_remainder_inplace(float* __restrict__ data,
                                                        float combined_factor,
                                                        int start_idx, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + start_idx;
    if (idx < size) {
        data[idx] *= combined_factor;
    }
}

torch::Tensor fused_scale_residual_inplace_hip(torch::Tensor input, float scaling_factor) {
    auto size = input.numel();
    
    // Combined factor: scaling_factor + 1 (since x * sf + x = x * (sf + 1))
    float combined_factor = scaling_factor + 1.0f;
    
    int num_float4 = size / 4;
    int remainder = size % 4;
    
    const int block_size = 1024;  // Maximize occupancy
    
    if (num_float4 > 0) {
        int num_blocks = min((num_float4 + block_size - 1) / block_size, 65535);
        
        fused_scale_residual_inplace_kernel<<<num_blocks, block_size>>>(
            reinterpret_cast<float4*>(input.data_ptr<float>()),
            combined_factor,
            num_float4
        );
    }
    
    if (remainder > 0) {
        int start_idx = num_float4 * 4;
        fused_scale_residual_remainder_inplace<<<1, 32>>>(
            input.data_ptr<float>(),
            combined_factor,
            start_idx, size
        );
    }
    
    return input;  // Return same tensor (in-place modification)
}
"""

fused_scale_residual_cpp = """
torch::Tensor fused_scale_residual_inplace_hip(torch::Tensor input, float scaling_factor);
"""

fused_module = load_inline(
    name="fused_scale_residual_v3",
    cpp_sources=fused_scale_residual_cpp,
    cuda_sources=fused_scale_residual_source,
    functions=["fused_scale_residual_inplace_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses scaling and residual addition in-place.
    
    x * scaling_factor + x = x * (1 + scaling_factor)
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Standard linear transformation
        x = self.matmul(x)
        # Fused in-place scaling + residual: x * sf + x = x * (1 + sf)
        x = fused_module.fused_scale_residual_inplace_hip(x, self.scaling_factor)
        return x
