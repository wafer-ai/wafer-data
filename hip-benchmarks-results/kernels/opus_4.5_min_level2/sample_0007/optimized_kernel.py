import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused Swish + Scaling kernel
swish_scale_cpp_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>

__global__ void swish_scale_kernel(const float* __restrict__ input, 
                                    float* __restrict__ output, 
                                    const float scaling_factor,
                                    const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process 4 elements per thread for better memory throughput
    int idx4 = idx * 4;
    
    if (idx4 + 3 < size) {
        // Load 4 floats at once using float4
        float4 in_val = *reinterpret_cast<const float4*>(input + idx4);
        float4 out_val;
        
        // Swish activation: x * sigmoid(x) * scaling_factor
        // sigmoid(x) = 1 / (1 + exp(-x))
        out_val.x = in_val.x / (1.0f + expf(-in_val.x)) * scaling_factor;
        out_val.y = in_val.y / (1.0f + expf(-in_val.y)) * scaling_factor;
        out_val.z = in_val.z / (1.0f + expf(-in_val.z)) * scaling_factor;
        out_val.w = in_val.w / (1.0f + expf(-in_val.w)) * scaling_factor;
        
        *reinterpret_cast<float4*>(output + idx4) = out_val;
    } else if (idx4 < size) {
        // Handle remaining elements
        for (int i = idx4; i < size && i < idx4 + 4; i++) {
            float val = input[i];
            output[i] = val / (1.0f + expf(-val)) * scaling_factor;
        }
    }
}

torch::Tensor swish_scale_hip(torch::Tensor input, float scaling_factor) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    // Each thread processes 4 elements
    const int num_blocks = (size / 4 + block_size - 1) / block_size;
    
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
    name="swish_scale",
    cpp_sources=swish_scale_cpp_decl,
    cuda_sources=swish_scale_cpp_source,
    functions=["swish_scale_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
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
