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

__global__ void sum_reduce_dim1_kernel_v3(
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
    int stride = inner_dim;
    
    // Sum over reduce_dim with strided access
    // Use multiple accumulators to hide latency
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;
    
    // Process 8 elements at a time
    int i = 0;
    int limit = reduce_dim - 7;
    for (; i < limit; i += 8) {
        sum0 += input[base_offset + i * stride];
        sum1 += input[base_offset + (i + 1) * stride];
        sum2 += input[base_offset + (i + 2) * stride];
        sum3 += input[base_offset + (i + 3) * stride];
        sum4 += input[base_offset + (i + 4) * stride];
        sum5 += input[base_offset + (i + 5) * stride];
        sum6 += input[base_offset + (i + 6) * stride];
        sum7 += input[base_offset + (i + 7) * stride];
    }
    
    // Handle remaining elements
    float remaining = 0.0f;
    for (; i < reduce_dim; i++) {
        remaining += input[base_offset + i * stride];
    }
    
    output[out_idx] = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 + remaining;
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
    
    sum_reduce_dim1_kernel_v3<<<num_blocks, BLOCK_SIZE>>>(
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
