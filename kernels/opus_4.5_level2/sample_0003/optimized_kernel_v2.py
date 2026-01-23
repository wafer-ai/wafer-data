import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel for scaling + residual add with better vectorization
fused_scale_residual_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Optimized kernel using float4 vectorization for maximum memory bandwidth
__global__ void fused_scale_residual_kernel_v2(const float4* __restrict__ input, 
                                                float4* __restrict__ output,
                                                float combined_factor,
                                                int num_float4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop for better occupancy
    for (int i = idx; i < num_float4; i += blockDim.x * gridDim.x) {
        float4 in_val = input[i];
        float4 out_val;
        out_val.x = in_val.x * combined_factor;
        out_val.y = in_val.y * combined_factor;
        out_val.z = in_val.z * combined_factor;
        out_val.w = in_val.w * combined_factor;
        output[i] = out_val;
    }
}

// Handle remainder elements
__global__ void fused_scale_residual_remainder(const float* __restrict__ input, 
                                                float* __restrict__ output,
                                                float combined_factor,
                                                int start_idx, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + start_idx;
    if (idx < size) {
        output[idx] = input[idx] * combined_factor;
    }
}

torch::Tensor fused_scale_residual_hip(torch::Tensor input, float scaling_factor) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    // Combined factor: scaling_factor + 1 (since x * sf + x = x * (sf + 1))
    float combined_factor = scaling_factor + 1.0f;
    
    int num_float4 = size / 4;
    int remainder = size % 4;
    
    const int block_size = 512;
    
    if (num_float4 > 0) {
        // Use many blocks for high occupancy on MI300X
        int num_blocks = min((num_float4 + block_size - 1) / block_size, 65535);
        
        fused_scale_residual_kernel_v2<<<num_blocks, block_size>>>(
            reinterpret_cast<const float4*>(input.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            combined_factor,
            num_float4
        );
    }
    
    if (remainder > 0) {
        int start_idx = num_float4 * 4;
        fused_scale_residual_remainder<<<1, remainder>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            combined_factor,
            start_idx, size
        );
    }
    
    return output;
}
"""

fused_scale_residual_cpp = """
torch::Tensor fused_scale_residual_hip(torch::Tensor input, float scaling_factor);
"""

fused_module = load_inline(
    name="fused_scale_residual_v2",
    cpp_sources=fused_scale_residual_cpp,
    cuda_sources=fused_scale_residual_source,
    functions=["fused_scale_residual_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses scaling and residual addition.
    
    x * scaling_factor + x = x * (1 + scaling_factor)
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Standard linear transformation
        x = self.matmul(x)
        # Fused scaling + residual: x * sf + x = x * (1 + sf)
        x = fused_module.fused_scale_residual_hip(x, self.scaling_factor)
        return x
