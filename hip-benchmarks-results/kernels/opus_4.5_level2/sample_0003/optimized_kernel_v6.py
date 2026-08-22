import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Try using mul_ in-place with the combined factor to minimize overhead
# Key insight: The reference creates a clone, then scales, then adds
# We can do: x * (1 + scaling_factor) directly, which is a single mul

fused_scale_residual_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Highly optimized for MI300X memory bandwidth
// Using float4 vectorization with optimal grid size

__global__ void fused_scale_kernel_mi300x(float4* __restrict__ data,
                                           const float combined_factor,
                                           const int num_float4) {
    // Use grid-stride loop pattern for maximum flexibility
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    
    while (idx < num_float4) {
        float4 val = data[idx];
        val.x *= combined_factor;
        val.y *= combined_factor;
        val.z *= combined_factor;
        val.w *= combined_factor;
        data[idx] = val;
        idx += stride;
    }
}

void fused_scale_inplace_hip(torch::Tensor input, float scaling_factor) {
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    
    const int size = input.numel();
    
    // Combined factor: x * sf + x = x * (sf + 1)
    const float combined_factor = scaling_factor + 1.0f;
    
    const int num_float4 = size / 4;
    
    if (num_float4 > 0) {
        // Maximize GPU utilization
        const int block_size = 256;
        const int num_blocks = min((num_float4 + block_size - 1) / block_size, 2048);
        
        fused_scale_kernel_mi300x<<<num_blocks, block_size>>>(
            reinterpret_cast<float4*>(input.data_ptr<float>()),
            combined_factor,
            num_float4
        );
    }
    
    // Handle remaining elements (if any) with torch
    const int remainder_start = num_float4 * 4;
    if (remainder_start < size) {
        auto remainder = input.slice(0, remainder_start, size);
        remainder.mul_(combined_factor);
    }
}
"""

fused_scale_residual_cpp = """
void fused_scale_inplace_hip(torch::Tensor input, float scaling_factor);
"""

fused_module = load_inline(
    name="fused_scale_v6",
    cpp_sources=fused_scale_residual_cpp,
    cuda_sources=fused_scale_residual_source,
    functions=["fused_scale_inplace_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses scaling and residual addition in-place.
    
    Original: x = matmul(x); orig = x.clone(); x = x * sf + orig
    Optimized: x = matmul(x); x *= (1 + sf)  (in-place)
    
    Saves: 1 clone, 1 detach, 1 add operation
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Standard linear transformation
        x = self.matmul(x)
        # Fused in-place scaling + residual: x * sf + x = x * (1 + sf)
        fused_module.fused_scale_inplace_hip(x, self.scaling_factor)
        return x
