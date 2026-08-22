import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

sum_reduce_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Two-pass reduction for better memory coalescing
// Input shape: (batch_size, reduce_dim, inner_dim)
// Output shape: (batch_size, 1, inner_dim)

// For MI300X, warp size is 64
#define WARP_SIZE 64

// Warp-level reduction using shuffle
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Block-level reduction
template<int BLOCK_SIZE>
__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];
    
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warpReduceSum(val);
    
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    
    // Only first warp does final reduction
    val = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? shared[lane] : 0.0f;
    
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    
    return val;
}

// Kernel that processes tiles of data for better coalescing
// Each block processes one (batch_idx, inner_chunk) and reduces over reduce_dim
template<int BLOCK_SIZE>
__global__ void sum_reduce_dim1_tiled(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int reduce_dim,
    int inner_dim
) {
    // blockIdx.x = batch_idx * num_inner_chunks + inner_chunk_idx
    int batch_idx = blockIdx.x / ((inner_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int inner_chunk_idx = blockIdx.x % ((inner_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int inner_idx = inner_chunk_idx * BLOCK_SIZE + threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    float sum = 0.0f;
    
    if (inner_idx < inner_dim) {
        // Calculate base input offset
        int base_offset = batch_idx * reduce_dim * inner_dim + inner_idx;
        int stride = inner_dim;
        
        // Sum over reduce_dim
        for (int i = 0; i < reduce_dim; i++) {
            sum += input[base_offset + i * stride];
        }
        
        // Write output
        output[batch_idx * inner_dim + inner_idx] = sum;
    }
}

// Alternative: Process in row-major order with coalesced reads
// Each warp cooperates to reduce one row
__global__ void sum_reduce_coalesced(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int reduce_dim,
    int inner_dim
) {
    // Each thread handles one output element
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * inner_dim;
    
    if (out_idx >= total_outputs) return;
    
    int batch_idx = out_idx / inner_dim;
    int inner_idx = out_idx % inner_dim;
    
    // Row start in input
    const float* row_start = input + batch_idx * reduce_dim * inner_dim + inner_idx;
    
    // Use inline asm-free reduction
    float sum = 0.0f;
    
    #pragma unroll 16
    for (int i = 0; i < reduce_dim; i++) {
        sum += row_start[i * inner_dim];
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
    const int BLOCK_SIZE = 512;
    int num_blocks = (total_outputs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    sum_reduce_coalesced<<<num_blocks, BLOCK_SIZE>>>(
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
