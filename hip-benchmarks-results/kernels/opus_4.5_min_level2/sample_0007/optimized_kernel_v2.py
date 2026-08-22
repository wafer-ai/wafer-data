import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized fused Swish + Scaling kernel for MI300X
swish_scale_cpp_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>

// Use fast sigmoid approximation and vectorized loads
__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__global__ void swish_scale_kernel_v2(const float* __restrict__ input, 
                                       float* __restrict__ output, 
                                       const float scaling_factor,
                                       const int size) {
    // Use more threads per block and more elements per thread
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple float4's
    for (int i = tid * 4; i < size; i += stride * 4) {
        if (i + 3 < size) {
            float4 in_val = *reinterpret_cast<const float4*>(input + i);
            float4 out_val;
            
            // Swish: x * sigmoid(x) * scale
            out_val.x = in_val.x * fast_sigmoid(in_val.x) * scaling_factor;
            out_val.y = in_val.y * fast_sigmoid(in_val.y) * scaling_factor;
            out_val.z = in_val.z * fast_sigmoid(in_val.z) * scaling_factor;
            out_val.w = in_val.w * fast_sigmoid(in_val.w) * scaling_factor;
            
            *reinterpret_cast<float4*>(output + i) = out_val;
        } else {
            // Handle remaining elements
            for (int j = i; j < size; j++) {
                float val = input[j];
                output[j] = val * fast_sigmoid(val) * scaling_factor;
            }
        }
    }
}

torch::Tensor swish_scale_hip(torch::Tensor input, float scaling_factor) {
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    // Optimize for MI300X: large number of threads
    const int block_size = 256;
    // Account for float4 processing
    const int num_elements_per_block = block_size * 4;
    int num_blocks = (size + num_elements_per_block - 1) / num_elements_per_block;
    // Cap number of blocks to maximize occupancy
    num_blocks = std::min(num_blocks, 65536);
    
    swish_scale_kernel_v2<<<num_blocks, block_size>>>(
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
    name="swish_scale_v2",
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
