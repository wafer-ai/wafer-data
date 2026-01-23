import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused SiLU (Swish) + Scaling kernel optimized for MI300X
# Uses aggressive vectorization and LDS for better performance
silu_scale_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Optimized fused SiLU + scale kernel using float4 vectorization
__global__ void silu_scale_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output, 
    const float scale,
    const int total_elements) 
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int vec_count = total_elements >> 2;  // total / 4
    
    // Process 4 elements at a time
    for (int i = tid; i < vec_count; i += stride) {
        int base_idx = i << 2;  // i * 4
        
        // Load 4 floats as float4
        float4 in = *reinterpret_cast<const float4*>(input + base_idx);
        float4 out;
        
        // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
        // Fused with scaling: x * sigmoid(x) * scale
        out.x = in.x * scale / (1.0f + expf(-in.x));
        out.y = in.y * scale / (1.0f + expf(-in.y));
        out.z = in.z * scale / (1.0f + expf(-in.z));
        out.w = in.w * scale / (1.0f + expf(-in.w));
        
        // Store 4 floats
        *reinterpret_cast<float4*>(output + base_idx) = out;
    }
    
    // Handle tail elements
    int remaining_start = vec_count << 2;
    for (int i = remaining_start + tid; i < total_elements; i += stride) {
        float x = input[i];
        output[i] = x * scale / (1.0f + expf(-x));
    }
}

torch::Tensor silu_scale_hip(torch::Tensor input, double scale) {
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on GPU");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    
    auto output = torch::empty_like(input);
    int total_elements = input.numel();
    
    // MI300X has 110 CUs, optimal block/grid sizing
    const int block_size = 256;
    const int max_blocks = 65535;
    int num_blocks = min(max_blocks, (total_elements / 4 + block_size - 1) / block_size);
    num_blocks = max(num_blocks, 1);
    
    silu_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        static_cast<float>(scale),
        total_elements
    );
    
    return output;
}
"""

silu_scale_cpp = """
torch::Tensor silu_scale_hip(torch::Tensor input, double scale);
"""

silu_scale_module = load_inline(
    name="silu_scale",
    cpp_sources=silu_scale_cpp,
    cuda_sources=silu_scale_source,
    functions=["silu_scale_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model with fused SiLU (Swish) + scaling kernel.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.silu_scale = silu_scale_module

    def forward(self, x):
        x = self.matmul(x)
        x = self.silu_scale.silu_scale_hip(x, self.scaling_factor)
        return x
