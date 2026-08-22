import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused Swish + Scaling kernel with more aggressive optimizations
swish_scale_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Use vectorized loads for better memory bandwidth utilization
__global__ void swish_scale_kernel_v2(const float* __restrict__ input, 
                                       float* __restrict__ output, 
                                       const float scaling_factor,
                                       const int size) {
    // Calculate global thread index
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    
    // Process 4 elements per iteration with vectorized loads
    for (int idx = tid * 4; idx < size; idx += stride * 4) {
        if (idx + 3 < size) {
            // Vectorized load
            float4 in_val = *reinterpret_cast<const float4*>(input + idx);
            float4 out_val;
            
            // Fast sigmoid approximation and swish computation
            // Using __expf for faster computation
            float sigmoid0 = __frcp_rn(1.0f + __expf(-in_val.x));
            float sigmoid1 = __frcp_rn(1.0f + __expf(-in_val.y));
            float sigmoid2 = __frcp_rn(1.0f + __expf(-in_val.z));
            float sigmoid3 = __frcp_rn(1.0f + __expf(-in_val.w));
            
            out_val.x = in_val.x * sigmoid0 * scaling_factor;
            out_val.y = in_val.y * sigmoid1 * scaling_factor;
            out_val.z = in_val.z * sigmoid2 * scaling_factor;
            out_val.w = in_val.w * sigmoid3 * scaling_factor;
            
            // Vectorized store
            *reinterpret_cast<float4*>(output + idx) = out_val;
        }
        else {
            // Handle remaining elements
            for (int i = idx; i < size; i++) {
                float val = input[i];
                float sigmoid_val = __frcp_rn(1.0f + __expf(-val));
                output[i] = val * sigmoid_val * scaling_factor;
            }
        }
    }
}

torch::Tensor swish_scale_hip(torch::Tensor input, float scaling_factor) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    // Optimize block and grid size for MI300X
    const int block_size = 512;
    // Each thread processes 4 elements, maximize occupancy
    const int elements_per_block = block_size * 4;
    const int num_blocks = min(65535, (size + elements_per_block - 1) / elements_per_block);
    
    swish_scale_kernel_v2<<<num_blocks, block_size>>>(
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
    name="swish_scale_v2",
    cpp_sources=swish_scale_cpp,
    cuda_sources=swish_scale_source,
    functions=["swish_scale_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math", "--gpu-max-threads-per-block=512"]
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
