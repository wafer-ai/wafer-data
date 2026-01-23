import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Custom HIP kernel for sum reduction over dimension 1
sum_reduction_hip_code = """
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Optimized kernel for sum reduction over dimension 1
// Input shape: [batch_size, reduce_dim, dim2]
// Output shape: [batch_size, 1, dim2]
__global__ void sum_reduction_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int reduce_dim,
    int dim2
) {
    // Each thread block handles one batch element
    int batch_idx = blockIdx.x;
    int dim2_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || dim2_idx >= dim2) return;
    
    // Shared memory for partial sums within a block
    __shared__ float shared_mem[256];
    
    int tid = threadIdx.x;
    float sum = 0.0f;
    
    // Grid-stride loop over reduction dimension
    // Each thread sums multiple elements to amortize overhead
    for (int i = tid; i < reduce_dim; i += blockDim.x) {
        int idx = batch_idx * (reduce_dim * dim2) + i * dim2 + dim2_idx;
        sum += input[idx];
    }
    
    // Store thread's partial sum in shared memory
    shared_mem[tid] = sum;
    __syncthreads();
    
    // Warp-level reduction using butterfly pattern
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared_mem[tid] += shared_mem[tid + offset];
        }
        __syncthreads();
    }
    
    // Thread 0 writes the final result for this block
    if (tid == 0) {
        int out_idx = batch_idx * dim2 + dim2_idx;
        output[out_idx] = shared_mem[0];
    }
}

torch::Tensor sum_reduction_hip(torch::Tensor input, int dim) {
    // Input is 3D: [batch_size, reduce_dim, dim2]
    auto sizes = input.sizes();
    int batch_size = sizes[0];
    int reduce_dim = sizes[1];
    int dim2 = sizes[2];
    
    // Allocate output tensor with shape [batch_size, 1, dim2]
    auto output_sizes = input.sizes().vec();
    output_sizes[1] = 1;
    auto output = torch::zeros(output_sizes, input.options());
    
    // Launch kernel with 2D grid
    // Grid: (batch_size, dim2)
    // Block: 256 threads
    dim3 grid(batch_size, dim2);
    dim3 block(256);
    
    hipLaunchKernelGGL(
        sum_reduction_kernel,
        grid, block,
        0, 0,
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        reduce_dim,
        dim2
    );
    
    return output;
}
"""

# Compile the HIP kernel
sum_reduction_hip = load_inline(
    name="sum_reduction",
    cpp_sources=sum_reduction_hip_code,
    functions=["sum_reduction_hip"],
    verbose=True,
)

batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1

class ModelNew(nn.Module):
    """
    Optimized model that performs sum reduction over a specified dimension using custom HIP kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.sum_reduction_hip = sum_reduction_hip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies sum reduction over the specified dimension using optimized HIP kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        # Use custom HIP kernel for dimension 1 reduction
        if self.dim == 1 and x.dim() == 3:
            return self.sum_reduction_hip.sum_reduction_hip(x, self.dim)
        else:
            # Fall back to PyTorch for other cases
            return torch.sum(x, dim=self.dim, keepdim=True)

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [reduce_dim]