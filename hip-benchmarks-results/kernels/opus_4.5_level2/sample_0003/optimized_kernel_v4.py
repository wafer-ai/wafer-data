import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Simple but optimized kernel using mul_add fusion
# Since x * sf + x = x * (sf + 1), we just need a simple multiply
fused_scale_residual_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Use larger vector types for better memory bandwidth on MI300X
// Process 8 floats at a time using two float4s per thread

__global__ void fused_scale_kernel(const float* __restrict__ input, 
                                    float* __restrict__ output,
                                    float combined_factor,
                                    int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    
    if (idx + 7 < size) {
        // Load 8 floats as 2 float4s
        float4 val0 = *reinterpret_cast<const float4*>(input + idx);
        float4 val1 = *reinterpret_cast<const float4*>(input + idx + 4);
        
        // Multiply
        val0.x *= combined_factor;
        val0.y *= combined_factor;
        val0.z *= combined_factor;
        val0.w *= combined_factor;
        val1.x *= combined_factor;
        val1.y *= combined_factor;
        val1.z *= combined_factor;
        val1.w *= combined_factor;
        
        // Store
        *reinterpret_cast<float4*>(output + idx) = val0;
        *reinterpret_cast<float4*>(output + idx + 4) = val1;
    } else {
        // Handle boundary cases
        for (int i = idx; i < min(idx + 8, size); i++) {
            output[i] = input[i] * combined_factor;
        }
    }
}

torch::Tensor fused_scale_hip(torch::Tensor input, float scaling_factor) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    // Combined factor: x * sf + x = x * (sf + 1)
    float combined_factor = scaling_factor + 1.0f;
    
    const int block_size = 256;
    // Each thread handles 8 elements
    const int num_blocks = (size + block_size * 8 - 1) / (block_size * 8);
    
    fused_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        combined_factor,
        size
    );
    
    return output;
}
"""

fused_scale_residual_cpp = """
torch::Tensor fused_scale_hip(torch::Tensor input, float scaling_factor);
"""

fused_module = load_inline(
    name="fused_scale_v4",
    cpp_sources=fused_scale_residual_cpp,
    cuda_sources=fused_scale_residual_source,
    functions=["fused_scale_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses scaling and residual addition.
    
    x * scaling_factor + x = x * (1 + scaling_factor)
    
    Key optimization: eliminate clone and fuse scale + add
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Standard linear transformation
        x = self.matmul(x)
        # Fused scaling + residual: x * sf + x = x * (1 + sf)
        x = fused_module.fused_scale_hip(x, self.scaling_factor)
        return x
