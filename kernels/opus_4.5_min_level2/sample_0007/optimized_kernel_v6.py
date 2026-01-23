import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused SiLU (Swish) + Scaling using rocBLAS for matmul
# Key insight: We can use rocm's built-in ops but fuse the activation

fused_silu_scale_cpp_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>

// Highly optimized kernel for SiLU + scale
// Uses larger block size and more aggressive vectorization
__global__ void fused_silu_scale_kernel(const float* __restrict__ input, 
                                         float* __restrict__ output, 
                                         const float scaling_factor,
                                         const int64_t size) {
    // Each thread processes 8 elements using 2 float4
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t base_idx = tid * 8;
    
    if (base_idx + 7 < size) {
        // Load 8 floats (2 x float4)
        float4 in1 = *reinterpret_cast<const float4*>(input + base_idx);
        float4 in2 = *reinterpret_cast<const float4*>(input + base_idx + 4);
        
        // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
        // Apply scaling factor
        in1.x = in1.x / (1.0f + __expf(-in1.x)) * scaling_factor;
        in1.y = in1.y / (1.0f + __expf(-in1.y)) * scaling_factor;
        in1.z = in1.z / (1.0f + __expf(-in1.z)) * scaling_factor;
        in1.w = in1.w / (1.0f + __expf(-in1.w)) * scaling_factor;
        
        in2.x = in2.x / (1.0f + __expf(-in2.x)) * scaling_factor;
        in2.y = in2.y / (1.0f + __expf(-in2.y)) * scaling_factor;
        in2.z = in2.z / (1.0f + __expf(-in2.z)) * scaling_factor;
        in2.w = in2.w / (1.0f + __expf(-in2.w)) * scaling_factor;
        
        // Store results
        *reinterpret_cast<float4*>(output + base_idx) = in1;
        *reinterpret_cast<float4*>(output + base_idx + 4) = in2;
    } else if (base_idx < size) {
        // Handle remaining elements
        for (int64_t i = base_idx; i < size; i++) {
            float x = input[i];
            output[i] = x / (1.0f + __expf(-x)) * scaling_factor;
        }
    }
}

torch::Tensor fused_silu_scale_hip(torch::Tensor input, float scaling_factor) {
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    // Each thread processes 8 elements
    const int64_t num_threads_needed = (size + 7) / 8;
    const int num_blocks = (num_threads_needed + block_size - 1) / block_size;
    
    fused_silu_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        scaling_factor,
        size
    );
    
    return output;
}
"""

fused_silu_scale_cpp_decl = """
torch::Tensor fused_silu_scale_hip(torch::Tensor input, float scaling_factor);
"""

fused_silu_scale_module = load_inline(
    name="fused_silu_scale_v6",
    cpp_sources=fused_silu_scale_cpp_decl,
    cuda_sources=fused_silu_scale_cpp_source,
    functions=["fused_silu_scale_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model with fused SiLU + scaling
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.matmul(x)
        # Fused SiLU + scaling in single kernel
        x = fused_silu_scale_module.fused_silu_scale_hip(x, self.scaling_factor)
        return x
