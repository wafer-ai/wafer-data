import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

sum_reduce_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Optimized sum reduction kernel - use vectorized loads and process multiple outputs per block
// Input shape: (batch_size, reduce_dim, inner_dim)
// Output shape: (batch_size, 1, inner_dim)

__global__ void sum_reduce_dim1_kernel_v2(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int reduce_dim,
    int inner_dim,
    int total_outputs
) {
    // Each thread handles one output element (one batch, inner pair)
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_idx >= total_outputs) return;
    
    int batch_idx = out_idx / inner_dim;
    int inner_idx = out_idx % inner_dim;
    
    // Calculate base input offset
    int base_offset = batch_idx * reduce_dim * inner_dim + inner_idx;
    
    // Sum over reduce_dim with strided access
    float sum = 0.0f;
    
    // Process 4 elements at a time when possible
    int i = 0;
    for (; i + 3 < reduce_dim; i += 4) {
        sum += input[base_offset + i * inner_dim];
        sum += input[base_offset + (i + 1) * inner_dim];
        sum += input[base_offset + (i + 2) * inner_dim];
        sum += input[base_offset + (i + 3) * inner_dim];
    }
    
    // Handle remaining elements
    for (; i < reduce_dim; i++) {
        sum += input[base_offset + i * inner_dim];
    }
    
    output[out_idx] = sum;
}

torch::Tensor sum_reduce_dim1_hip(torch::Tensor input, int dim) {
    auto batch_size = input.size(0);
    auto reduce_dim = input.size(1);
    auto inner_dim = input.size(2);
    
    // Output shape: (batch_size, 1, inner_dim)
    auto output = torch::empty({batch_size, 1, inner_dim}, input.options());
    
    int total_outputs = batch_size * inner_dim;
    const int BLOCK_SIZE = 256;
    int num_blocks = (total_outputs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    sum_reduce_dim1_kernel_v2<<<num_blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        reduce_dim,
        inner_dim,
        total_outputs
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
