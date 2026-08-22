import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

gelu_cpp_source = """
#include <hip/hip_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* __restrict__ x, float* __restrict__ out, int size) {
    // Each block processes a contiguous chunk of memory for better cache locality
    int block_start = blockIdx.x * blockDim.x * 8;
    int stride = blockDim.x;
    
    // GELU constants
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coefficient = 0.044715f;
    
    // Process 8 elements per thread with stride
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = block_start + threadIdx.x * 8 + i;
        if (idx < size) {
            float xi = x[idx];
            float x3 = xi * xi * xi;
            float arg = sqrt_2_over_pi * (xi + coefficient * x3);
            float tanh_val = tanhf(arg);
            out[idx] = 0.5f * xi * (1.0f + tanh_val);
        }
    }
}

torch::Tensor gelu_hip(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    // Use 256 threads per block, each processing 8 elements
    const int block_size = 256;
    const int elements_per_thread = 8;
    const int num_blocks = (size + block_size * elements_per_thread - 1) / (block_size * elements_per_thread);

    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

gelu_module = load_inline(
    name="gelu_optimized_256",
    cpp_sources=gelu_cpp_source,
    functions=["gelu_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Simple model that performs a GELU activation with custom HIP kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu_hip = gelu_module.gelu_hip
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies GELU activation to the input tensor using custom HIP kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        return self.gelu_hip(x)