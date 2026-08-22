import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>

__global__ void sum_reduce_dim1_kernel(const float* __restrict__ input, 
                                       float* __restrict__ output, 
                                       int B, int D1, int D2) {
    long long d2 = blockIdx.x * blockDim.x + threadIdx.x;
    long long b = blockIdx.y;
    
    if (d2 < D2 && b < B) {
        long long stride = D2;
        // Use 64-bit index arithmetic to handle potentially large tensors
        long long input_idx = b * (long long)D1 * D2 + d2;
        
        float sum = 0.0f;
        
        #pragma unroll 4
        for (int k = 0; k < D1; ++k) {
            sum += input[input_idx];
            input_idx += stride;
        }
        
        output[b * (long long)D2 + d2] = sum;
    }
}

torch::Tensor sum_reduce_dim1(torch::Tensor input) {
    input = input.contiguous();
    
    int B = input.size(0);
    int D1 = input.size(1);
    int D2 = input.size(2);
    
    auto output = torch::empty({B, 1, D2}, input.options());
    
    int threads_per_block = 256;
    int blocks_x = (D2 + threads_per_block - 1) / threads_per_block;
    int blocks_y = B;
    
    dim3 blocks(blocks_x, blocks_y);
    dim3 threads(threads_per_block);
    
    // Launch on default stream (0) for simplicity and implicit synchronization
    sum_reduce_dim1_kernel<<<blocks, threads, 0, 0>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2
    );
    
    return output;
}
"""

module = load_inline(
    name="sum_reduce_kernels_final",
    cpp_sources=cpp_source,
    functions=["sum_reduce_dim1"],
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.sum_reduce = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optimized path for dim=1 and 3D input
        if self.dim == 1 and x.dim() == 3:
            return self.sum_reduce.sum_reduce_dim1(x)
        return torch.sum(x, dim=self.dim, keepdim=True)
