import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused Swish + Scaling kernel using grid-stride loops
swish_scale_cpp_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>

// Process multiple elements per thread using grid-stride loops
__global__ void swish_scale_kernel(const float* __restrict__ input, 
                                    float* __restrict__ output, 
                                    const float scaling_factor,
                                    const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop with vectorized access
    for (int i = idx; i < size / 4; i += stride) {
        const int base = i * 4;
        float4 val = *reinterpret_cast<const float4*>(input + base);
        
        // Compute swish: x * sigmoid(x) * scale
        val.x = val.x * (1.0f / (1.0f + __expf(-val.x))) * scaling_factor;
        val.y = val.y * (1.0f / (1.0f + __expf(-val.y))) * scaling_factor;
        val.z = val.z * (1.0f / (1.0f + __expf(-val.z))) * scaling_factor;
        val.w = val.w * (1.0f / (1.0f + __expf(-val.w))) * scaling_factor;
        
        *reinterpret_cast<float4*>(output + base) = val;
    }
    
    // Handle remainder
    const int remainder_start = (size / 4) * 4;
    for (int i = remainder_start + idx; i < size; i += stride) {
        float x = input[i];
        output[i] = x * (1.0f / (1.0f + __expf(-x))) * scaling_factor;
    }
}

torch::Tensor swish_scale_hip(torch::Tensor input, float scaling_factor) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 512;
    // Launch enough blocks to keep GPU busy but not too many
    const int num_blocks = std::min((int)((size / 4 + block_size - 1) / block_size), 1024);
    
    swish_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        scaling_factor,
        size
    );
    
    return output;
}
"""

swish_scale_cpp_decl = """
torch::Tensor swish_scale_hip(torch::Tensor input, float scaling_factor);
"""

swish_scale_module = load_inline(
    name="swish_scale_v5",
    cpp_sources=swish_scale_cpp_decl,
    cuda_sources=swish_scale_cpp_source,
    functions=["swish_scale_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math", "--gpu-max-threads-per-block=512"]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication, applies fused Swish activation + scaling.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.matmul(x)
        # Fused Swish + scaling
        x = swish_scale_module.swish_scale_hip(x, self.scaling_factor)
        return x
