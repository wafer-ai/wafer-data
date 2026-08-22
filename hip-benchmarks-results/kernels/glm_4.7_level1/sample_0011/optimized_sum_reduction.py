import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

sum_reduction_cpp_source = """
#include <hip/hip_runtime.h>

#define TILE_SIZE 16
#define BLOCK_SIZE 256

__global__ void sum_reduction_2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int reduce_dim,
    int other_dim
) {
    // Each thread block handles TILE_SIZE x TILE_SIZE elements
    int batch_start = blockIdx.x * TILE_SIZE;
    int other_start = blockIdx.y * TILE_SIZE;
    
    int tid = threadIdx.x;
    
    // Calculate which (batch, other) pair this thread handles
    int batch_thread = tid / TILE_SIZE;
    int other_thread = tid % TILE_SIZE;
    
    int batch_idx = batch_start + batch_thread;
    int other_idx = other_start + other_thread;
    
    // Shared memory for reduction - TILE_SIZE * TILE_SIZE elements
    __shared__ float s_data[TILE_SIZE * TILE_SIZE];
    
    float sum = 0.0f;
    
    if (batch_idx < batch_size && other_idx < other_dim) {
        size_t base = (size_t)batch_idx * reduce_dim * other_dim + other_idx;
        
        // Each thread processes the entire reduction dimension
        for (int i = 0; i < reduce_dim; i++) {
            sum += input[base + (size_t)i * other_dim];
        }
    }
    
    s_data[tid] = sum;
    __syncthreads();
    
    // Reduce within shared memory
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    
    // Write results - need to handle TILE_SIZE x TILE_SIZE output values
    // Each block writes TILE_SIZE * TILE_SIZE results
    if (tid < TILE_SIZE * TILE_SIZE) {
        int batch_out = batch_start + (tid / TILE_SIZE);
        int other_out = other_start + (tid % TILE_SIZE);
        
        if (batch_out < batch_size && other_out < other_dim) {
            output[batch_out * other_dim + other_out] = s_data[tid];
        }
    }
}

torch::Tensor sum_reduction_hip(torch::Tensor x, int dim) {
    TORCH_CHECK(x.dim() == 3, "Input tensor must be 3-dimensional");
    
    auto batch_size = x.size(0);
    auto dim1 = x.size(1);
    auto dim2 = x.size(2);
    
    auto output = torch::zeros({batch_size, 1, dim2}, x.options());
    
    int reduce_dim_size = dim1;
    int other_dim_size = dim2;
    
    // Calculate grid dimensions
    int grid_x = (batch_size + TILE_SIZE - 1) / TILE_SIZE;
    int grid_y = (other_dim_size + TILE_SIZE - 1) / TILE_SIZE;
    
    dim3 grid(grid_x, grid_y);
    dim3 block(BLOCK_SIZE);
    
    sum_reduction_2d_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        reduce_dim_size,
        other_dim_size
    );
    
    return output;
}
"""

sum_reduction = load_inline(
    name="sum_reduction",
    cpp_sources=sum_reduction_cpp_source,
    functions=["sum_reduction_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model with custom HIP kernel for sum reduction.
    """
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.sum_reduction = sum_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sum_reduction.sum_reduction_hip(x, self.dim)