
import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Set the compiler to hipcc to ensure ROCm headers are found
os.environ["CXX"] = "hipcc"

# Define the CUDA/HIP kernel source code
cpp_source = """
#include <hip/hip_runtime.h>

__global__ void fused_scale_add_kernel(float* __restrict__ x, float scale, int size) {
    // Calculate global thread index
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    // Precompute scale multiplier: x * s + x = x * (s + 1)
    float multiplier = scale + 1.0f;
    
    // Check if we can process a full float4 vector
    if (idx + 3 < size) {
        // Reinterpret pointer as float4 for vectorized load/store
        float4* ptr = reinterpret_cast<float4*>(&x[idx]);
        float4 v = *ptr;
        
        // Apply scaling
        v.x *= multiplier;
        v.y *= multiplier;
        v.z *= multiplier;
        v.w *= multiplier;
        
        // Write back
        *ptr = v;
    } else {
        // Handle remaining elements (tail case)
        for (int i = idx; i < size; ++i) {
            x[i] *= multiplier;
        }
    }
}

torch::Tensor fused_scale_add_hip(torch::Tensor x, float scale) {
    int size = x.numel();
    const int block_size = 256;
    // Calculate grid size to cover all elements with float4 (4 elements per thread)
    // Total threads needed = ceil(size / 4)
    int total_threads = (size + 3) / 4;
    const int num_blocks = (total_threads + block_size - 1) / block_size;
    
    fused_scale_add_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), scale, size);
    
    return x;
}
"""

# Compile and load the kernel
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=cpp_source,
    functions=["fused_scale_add_hip"],
    extra_cflags=['-O3'],
    verbose=True
)

class ModelNew(nn.Module):
    """
    Optimized model that replaces the element-wise operations with a custom HIP kernel.
    The Matmul is kept as is (using optimized rocBLAS), while the subsequent
    scale and residual add are fused into a single in-place kernel.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.fused_ops = fused_ops

    def forward(self, x):
        """
        Forward pass using custom fused kernel.
        Original logic:
            x = self.matmul(x)
            original_x = x.clone().detach()
            x = x * self.scaling_factor
            x = x + original_x
        
        Optimized logic:
            x = self.matmul(x)
            x *= (scaling_factor + 1)  [In-place fused kernel]
        """
        x = self.matmul(x)
        self.fused_ops.fused_scale_add_hip(x, self.scaling_factor)
        return x

def get_inputs():
    batch_size = 16384
    in_features = 4096
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    in_features = 4096
    out_features = 4096
    scaling_factor = 0.5
    return [in_features, out_features, scaling_factor]
