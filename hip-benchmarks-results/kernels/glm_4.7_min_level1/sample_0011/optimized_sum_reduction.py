import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

sum_reduction_cpp_source = """
#include <hip/hip_runtime.h>

#define BLOCK_SIZE 256

__global__ void sum_reduction_kernel(const float* input, float* output, int batch_size, int dim1, int dim2) {
    // Global output position
    int output_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    if (output_idx >= batch_size * dim2) {
        return;
    }
    
    // Calculate the output position (b, c)
    int b = output_idx / dim2;
    int c = output_idx % dim2;
    
    // Shared memory for parallel reduction within this thread's calculation
    extern __shared__ float shared_data[];
    
    int thread_idx = threadIdx.x;
    
    // Each thread reduces over dim1 independently
    // Load elements into shared memory then reduce
    float sum = 0.0f;
    int elements_per_thread = (dim1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int i = thread_idx; i < dim1; i += BLOCK_SIZE) {
        int linear_idx = (b * dim1 + i) * dim2 + c;
        sum += input[linear_idx];
    }
    
    shared_data[thread_idx] = sum;
    __syncthreads();
    
    // Tree reduction in shared memory for this thread's multiple loads
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (thread_idx < stride) {
            shared_data[thread_idx] += shared_data[thread_idx + stride];
        }
        __syncthreads();
    }
    
    // Write final result (first thread of the block writes for each element this block handles)
    int block_start_output = blockIdx.x * BLOCK_SIZE;
    int elements_in_block = min(BLOCK_SIZE, batch_size * dim2 - block_start_output);
    
    if (thread_idx < elements_in_block) {
        int output_linear_idx = block_start_output + thread_idx;
        output[output_linear_idx] = shared_data[thread_idx];
    }
}

torch::Tensor sum_reduction_hip(torch::Tensor x, int dim) {
    // Get input dimensions
    int batch_size = x.size(0);
    int dim1 = x.size(1);
    int dim2 = x.size(2);
    
    // Output shape is (batch_size, 1, dim2) -> flatten to (batch_size, dim2)
    auto output = torch::zeros({batch_size, dim2}, x.options());
    
    // Number of output elements
    int num_outputs = batch_size * dim2;
    
    // Calculate number of blocks needed
    int num_blocks = (num_outputs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Launch kernel
    dim3 grid(num_blocks);
    dim3 block(BLOCK_SIZE);
    int shared_mem_size = BLOCK_SIZE * sizeof(float);
    
    sum_reduction_kernel<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, dim1, dim2
    );
    
    // Reshape to (batch_size, 1, dim2)
    return output.view({batch_size, 1, dim2});
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
    Simple model that performs sum reduction over a specified dimension using optimized HIP kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over (currently supports dim=1 for 3D tensors).
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.sum_reduction = sum_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies sum reduction over the specified dimension using optimized HIP kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        return self.sum_reduction.sum_reduction_hip(x, self.dim)


def get_inputs():
    x = torch.rand(128, 4096, 4095).cuda()
    return [x]


def get_init_inputs():
    return [1]