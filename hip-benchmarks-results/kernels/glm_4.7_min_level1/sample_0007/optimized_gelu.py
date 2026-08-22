import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

gelu_cpp_source = """
#include <hip/hip_runtime.h>

// Optimized GELU kernel for MI300X with focus on memory bandwidth
__global__ void gelu_kernel(const float* __restrict__ x, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Strided loop for even load distribution
    const int stride = blockDim.x * gridDim.x;
    
    for (; idx < size; idx += stride) {
        float xi = x[idx];
        
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        const float a = 0.7978845608028654f;  // sqrt(2/pi)
        const float b = 0.044715f;
        
        float x3 = xi * xi * xi;
        float tanh_arg = a * (xi + b * x3);
        float tanh_val = tanhf(tanh_arg);
        
        out[idx] = 0.5f * xi * (1.0f + tanh_val);
    }
}

torch::Tensor gelu_hip(torch::Tensor x) {
    int size = x.numel();
    auto out = torch::zeros_like(x);
    
    // Optimize for MI300X: maximize wavefronts per CU
    const int block_size = 256;
    const int min_blocks_per_sm = 4;
    const int max_blocks = 65536 * 32;  // Large number to ensure full GPU utilization
    
    int num_blocks = (size + block_size - 1) / block_size;
    num_blocks = min(num_blocks, max_blocks);
    
    hipLaunchKernelGGL(gelu_kernel, dim3(num_blocks), dim3(block_size), 0, 0, 
                       x.data_ptr<float>(), out.data_ptr<float>(), size);
    
    return out;
}
"""

gelu_module = load_inline(
    name="gelu_module",
    cpp_sources=gelu_cpp_source,
    functions=["gelu_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs GELU activation using custom HIP kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu_hip = gelu_module.gelu_hip
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies GELU activation to the input tensor using optimized HIP kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        return self.gelu_hip(x)

# Helper function to create model instance
def create_model():
    return ModelNew()