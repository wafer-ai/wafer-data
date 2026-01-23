import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

sum_reduce_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Optimized sum reduction kernel for reducing over dim=1
// Input shape: (batch_size, reduce_dim, inner_dim)
// Output shape: (batch_size, 1, inner_dim)

template<int BLOCK_SIZE>
__global__ void sum_reduce_dim1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int reduce_dim,
    int inner_dim
) {
    __shared__ float shared_data[BLOCK_SIZE];
    
    // Each block handles one (batch, inner) pair
    int batch_idx = blockIdx.x;
    int inner_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || inner_idx >= inner_dim) return;
    
    int tid = threadIdx.x;
    
    // Calculate base input offset for this (batch, inner) pair
    // Input layout: batch * (reduce_dim * inner_dim) + reduce * inner_dim + inner
    int base_offset = batch_idx * reduce_dim * inner_dim + inner_idx;
    
    // Each thread accumulates multiple elements with strided access
    float sum = 0.0f;
    for (int i = tid; i < reduce_dim; i += BLOCK_SIZE) {
        sum += input[base_offset + i * inner_dim];
    }
    
    shared_data[tid] = sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    #pragma unroll
    for (int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    
    // Warp-level reduction (no sync needed within a warp)
    if (tid < 32) {
        volatile float* vsmem = shared_data;
        if (BLOCK_SIZE >= 64) vsmem[tid] += vsmem[tid + 32];
        if (BLOCK_SIZE >= 32) vsmem[tid] += vsmem[tid + 16];
        if (BLOCK_SIZE >= 16) vsmem[tid] += vsmem[tid + 8];
        if (BLOCK_SIZE >= 8) vsmem[tid] += vsmem[tid + 4];
        if (BLOCK_SIZE >= 4) vsmem[tid] += vsmem[tid + 2];
        if (BLOCK_SIZE >= 2) vsmem[tid] += vsmem[tid + 1];
    }
    
    // Write result
    if (tid == 0) {
        // Output layout: batch * (1 * inner_dim) + inner
        output[batch_idx * inner_dim + inner_idx] = shared_data[0];
    }
}

torch::Tensor sum_reduce_dim1_hip(torch::Tensor input, int dim) {
    auto batch_size = input.size(0);
    auto reduce_dim = input.size(1);
    auto inner_dim = input.size(2);
    
    // Output shape: (batch_size, 1, inner_dim)
    auto output = torch::empty({batch_size, 1, inner_dim}, input.options());
    
    const int BLOCK_SIZE = 256;
    dim3 grid(batch_size, inner_dim);
    dim3 block(BLOCK_SIZE);
    
    sum_reduce_dim1_kernel<BLOCK_SIZE><<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        reduce_dim,
        inner_dim
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor sum_reduce_dim1_hip(torch::Tensor input, int dim);
"""

sum_reduce_module = load_inline(
    name="sum_reduce",
    cpp_sources=cpp_source,
    cuda_sources=sum_reduce_hip_source,
    functions=["sum_reduce_dim1_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs sum reduction over a specified dimension using HIP kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.sum_reduce = sum_reduce_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies sum reduction over the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        # Use custom kernel for dim=1 reduction
        if self.dim == 1 and x.dim() == 3:
            return self.sum_reduce.sum_reduce_dim1_hip(x, self.dim)
        else:
            return torch.sum(x, dim=self.dim, keepdim=True)


def custom_kernel(inputs):
    x = inputs[0]
    model = ModelNew(dim=1)
    return model.forward(x)
