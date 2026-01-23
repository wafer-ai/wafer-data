import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel for scaling + residual add: output = x * scaling_factor + x = x * (1 + scaling_factor)
fused_scale_residual_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void fused_scale_residual_kernel(const float* __restrict__ input, 
                                             float* __restrict__ output,
                                             float combined_factor,
                                             int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process 4 elements per thread for better memory bandwidth
    int idx4 = idx * 4;
    
    if (idx4 + 3 < size) {
        // Vectorized load using float4
        float4 in_val = *reinterpret_cast<const float4*>(input + idx4);
        float4 out_val;
        out_val.x = in_val.x * combined_factor;
        out_val.y = in_val.y * combined_factor;
        out_val.z = in_val.z * combined_factor;
        out_val.w = in_val.w * combined_factor;
        *reinterpret_cast<float4*>(output + idx4) = out_val;
    } else if (idx4 < size) {
        // Handle remainder
        for (int i = idx4; i < size && i < idx4 + 4; i++) {
            output[i] = input[i] * combined_factor;
        }
    }
}

torch::Tensor fused_scale_residual_hip(torch::Tensor input, float scaling_factor) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    // Combined factor: scaling_factor + 1 (since x * sf + x = x * (sf + 1))
    float combined_factor = scaling_factor + 1.0f;
    
    const int block_size = 256;
    // Each thread processes 4 elements
    const int num_blocks = (size / 4 + block_size - 1) / block_size;
    
    fused_scale_residual_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        combined_factor,
        size
    );
    
    return output;
}
"""

fused_scale_residual_cpp = """
torch::Tensor fused_scale_residual_hip(torch::Tensor input, float scaling_factor);
"""

fused_module = load_inline(
    name="fused_scale_residual",
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
