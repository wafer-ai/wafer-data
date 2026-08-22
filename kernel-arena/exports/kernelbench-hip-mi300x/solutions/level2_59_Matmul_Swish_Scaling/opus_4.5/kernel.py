import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Ultra-optimized fused SiLU + Scaling kernel
# Process 8 elements per thread with maximum parallelism
silu_scale_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Process 8 floats per thread using two float4 loads
__global__ __launch_bounds__(256, 8)
void silu_scale_kernel_v3(
    const float* __restrict__ input, 
    float* __restrict__ output, 
    const float scale,
    const int n) 
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_stride = gridDim.x * blockDim.x;
    const int n8 = n >> 3;  // n / 8
    
    // Process 8 elements per iteration
    for (int i = tid; i < n8; i += grid_stride) {
        int base = i << 3;  // i * 8
        
        // Load 8 floats as two float4s
        float4 v0 = *reinterpret_cast<const float4*>(input + base);
        float4 v1 = *reinterpret_cast<const float4*>(input + base + 4);
        
        // Compute SiLU + scale
        #pragma unroll
        {
            v0.x = v0.x * scale / (1.0f + __expf(-v0.x));
            v0.y = v0.y * scale / (1.0f + __expf(-v0.y));
            v0.z = v0.z * scale / (1.0f + __expf(-v0.z));
            v0.w = v0.w * scale / (1.0f + __expf(-v0.w));
            v1.x = v1.x * scale / (1.0f + __expf(-v1.x));
            v1.y = v1.y * scale / (1.0f + __expf(-v1.y));
            v1.z = v1.z * scale / (1.0f + __expf(-v1.z));
            v1.w = v1.w * scale / (1.0f + __expf(-v1.w));
        }
        
        // Store 8 floats
        *reinterpret_cast<float4*>(output + base) = v0;
        *reinterpret_cast<float4*>(output + base + 4) = v1;
    }
    
    // Handle remainder
    int base = n8 << 3;
    for (int i = base + tid; i < n; i += grid_stride) {
        float x = input[i];
        output[i] = x * scale / (1.0f + __expf(-x));
    }
}

torch::Tensor silu_scale_hip(torch::Tensor input, double scale) {
    auto output = torch::empty_like(input);
    int n = input.numel();
    
    // Optimal configuration for MI300X
    const int block_size = 256;
    // MI300X has 110 CUs with 4 SIMDs each
    int num_blocks = (n / 8 + block_size - 1) / block_size;
    num_blocks = std::min(num_blocks, 65535);
    num_blocks = std::max(num_blocks, 1);
    
    silu_scale_kernel_v3<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        static_cast<float>(scale),
        n
    );
    
    return output;
}
"""

silu_scale_cpp = """
torch::Tensor silu_scale_hip(torch::Tensor input, double scale);
"""

silu_scale_module = load_inline(
    name="silu_scale_v3",
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
