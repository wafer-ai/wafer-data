import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused Swish + Scaling kernel with in-place operation
swish_scale_inplace_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// In-place kernel with maximum vectorization for MI300X
__global__ void swish_scale_inplace_kernel(
    float* __restrict__ data, 
    const float scaling_factor,
    const int size) 
{
    // Grid-stride loop for maximum occupancy
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    
    // Process 4 elements per thread using vectorized loads/stores
    const int vec_size = size / 4;
    
    for (int vec_idx = tid; vec_idx < vec_size; vec_idx += stride) {
        int idx = vec_idx * 4;
        
        // Vectorized load
        float4 val = *reinterpret_cast<float4*>(data + idx);
        
        // Compute swish and scale
        float sigmoid0 = 1.0f / (1.0f + expf(-val.x));
        float sigmoid1 = 1.0f / (1.0f + expf(-val.y));
        float sigmoid2 = 1.0f / (1.0f + expf(-val.z));
        float sigmoid3 = 1.0f / (1.0f + expf(-val.w));
        
        val.x = val.x * sigmoid0 * scaling_factor;
        val.y = val.y * sigmoid1 * scaling_factor;
        val.z = val.z * sigmoid2 * scaling_factor;
        val.w = val.w * sigmoid3 * scaling_factor;
        
        // Vectorized store
        *reinterpret_cast<float4*>(data + idx) = val;
    }
    
    // Handle remaining elements
    const int remaining_start = vec_size * 4;
    for (int idx = remaining_start + tid; idx < size; idx += stride) {
        float val = data[idx];
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        data[idx] = val * sigmoid_val * scaling_factor;
    }
}

void swish_scale_inplace_hip(torch::Tensor data, float scaling_factor) {
    auto size = data.numel();
    
    // Optimize for MI300X - use many threads and large grid
    const int block_size = 1024;
    const int num_blocks = min(65535, (size / 4 + block_size - 1) / block_size);
    
    swish_scale_inplace_kernel<<<num_blocks, block_size>>>(
        data.data_ptr<float>(), 
        scaling_factor,
        size
    );
}
"""

swish_scale_inplace_cpp = """
void swish_scale_inplace_hip(torch::Tensor data, float scaling_factor);
"""

swish_scale_module = load_inline(
    name="swish_scale_inplace",
    cpp_sources=swish_scale_inplace_cpp,
    cuda_sources=swish_scale_inplace_source,
    functions=["swish_scale_inplace_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model using in-place fused Swish + scaling kernel.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.swish_scale = swish_scale_module

    def forward(self, x):
        x = self.matmul(x)
        self.swish_scale.swish_scale_inplace_hip(x, self.scaling_factor)
        return x
