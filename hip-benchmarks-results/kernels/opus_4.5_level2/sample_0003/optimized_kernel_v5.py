import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized kernel with wave-level coalescing for MI300X
fused_scale_residual_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Use launch_bounds for MI300X to ensure high occupancy
// MI300X has 64 threads per wave (wavefront), targeting high occupancy

__global__ __launch_bounds__(256, 4)
void fused_scale_kernel_optimized(const float* __restrict__ input, 
                                   float* __restrict__ output,
                                   const float combined_factor,
                                   const int size) {
    // Calculate global thread index
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Process 4 elements per thread in a grid-stride loop
    // This ensures all memory accesses are coalesced
    for (int idx = tid * 4; idx < size; idx += stride * 4) {
        if (idx + 3 < size) {
            // Vectorized load/store
            float4 val = *reinterpret_cast<const float4*>(input + idx);
            val.x *= combined_factor;
            val.y *= combined_factor;
            val.z *= combined_factor;
            val.w *= combined_factor;
            *reinterpret_cast<float4*>(output + idx) = val;
        } else {
            // Handle remainder
            for (int i = idx; i < size && i < idx + 4; i++) {
                output[i] = input[i] * combined_factor;
            }
        }
    }
}

torch::Tensor fused_scale_hip(torch::Tensor input, float scaling_factor) {
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on GPU");
    
    const int size = input.numel();
    auto output = torch::empty_like(input);
    
    // Combined factor: x * sf + x = x * (sf + 1)
    const float combined_factor = scaling_factor + 1.0f;
    
    // MI300X has 304 CUs, aim for high occupancy
    // With 256 threads per block and 4 blocks per CU, we have 256*4*304 = 311296 threads
    // Each processing 4 elements = ~1.2M elements per iteration
    // Our size is 16384*4096 = 67M elements
    const int block_size = 256;
    const int num_blocks = min(1024, (size / 4 + block_size - 1) / block_size);
    
    fused_scale_kernel_optimized<<<num_blocks, block_size>>>(
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
    name="fused_scale_v5",
    cpp_sources=fused_scale_residual_cpp,
    cuda_sources=fused_scale_residual_source,
    functions=["fused_scale_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math", "-munsafe-fp-atomics"]
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
        x = fused_module.fused_scale_hip(x, self.scaling_factor)
        return x
