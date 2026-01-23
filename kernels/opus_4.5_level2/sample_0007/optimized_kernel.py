import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused Swish + Scaling kernel
swish_scale_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void swish_scale_kernel(const float* __restrict__ input, 
                                    float* __restrict__ output, 
                                    const float scaling_factor,
                                    const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process 4 elements per thread for better memory coalescing
    int idx4 = idx * 4;
    
    if (idx4 + 3 < size) {
        // Load 4 elements
        float4 in_val = *reinterpret_cast<const float4*>(input + idx4);
        float4 out_val;
        
        // Compute swish and scale for each element
        float sigmoid0 = 1.0f / (1.0f + expf(-in_val.x));
        out_val.x = in_val.x * sigmoid0 * scaling_factor;
        
        float sigmoid1 = 1.0f / (1.0f + expf(-in_val.y));
        out_val.y = in_val.y * sigmoid1 * scaling_factor;
        
        float sigmoid2 = 1.0f / (1.0f + expf(-in_val.z));
        out_val.z = in_val.z * sigmoid2 * scaling_factor;
        
        float sigmoid3 = 1.0f / (1.0f + expf(-in_val.w));
        out_val.w = in_val.w * sigmoid3 * scaling_factor;
        
        // Store 4 elements
        *reinterpret_cast<float4*>(output + idx4) = out_val;
    }
    else if (idx4 < size) {
        // Handle remaining elements
        for (int i = idx4; i < size && i < idx4 + 4; i++) {
            float val = input[i];
            float sigmoid_val = 1.0f / (1.0f + expf(-val));
            output[i] = val * sigmoid_val * scaling_factor;
        }
    }
}

torch::Tensor swish_scale_hip(torch::Tensor input, float scaling_factor) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    // Each thread processes 4 elements
    const int num_blocks = (size + block_size * 4 - 1) / (block_size * 4);
    
    swish_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        scaling_factor,
        size
    );
    
    return output;
}
"""

swish_scale_cpp = """
torch::Tensor swish_scale_hip(torch::Tensor input, float scaling_factor);
"""

swish_scale_module = load_inline(
    name="swish_scale",
    cpp_sources=swish_scale_cpp,
    cuda_sources=swish_scale_source,
    functions=["swish_scale_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model with fused Swish activation and scaling kernel.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.swish_scale = swish_scale_module

    def forward(self, x):
        x = self.matmul(x)
        x = self.swish_scale.swish_scale_hip(x, self.scaling_factor)
        return x
