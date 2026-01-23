import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

sum_reduction_cpp_source = """
torch::Tensor sum_reduction_hip(torch::Tensor input, int dim);
"""

sum_reduction_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Optimized sum reduction kernel for reducing over dim=1
// Input shape: (batch_size, reduce_dim, inner_dim)
// Output shape: (batch_size, 1, inner_dim)

#define BLOCK_SIZE 512
#define WARP_SIZE 64

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Process multiple output elements per block using 2D blocks
__global__ void sum_reduction_kernel_v2(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int reduce_dim,
    const int inner_dim
) {
    // Each block handles one output element
    int out_idx = blockIdx.x;
    int batch_idx = out_idx / inner_dim;
    int inner_idx = out_idx % inner_dim;
    
    if (batch_idx >= batch_size) return;
    
    // Base pointer for this (batch, *, inner) slice
    const float* input_slice = input + batch_idx * reduce_dim * inner_dim + inner_idx;
    
    // Each thread accumulates partial sum with loop unrolling
    float sum = 0.0f;
    
    int i = threadIdx.x;
    int stride = blockDim.x;
    
    // Unroll by 4 for better instruction-level parallelism
    int limit = reduce_dim - (reduce_dim % (stride * 4));
    for (; i < limit; i += stride * 4) {
        sum += input_slice[i * inner_dim];
        sum += input_slice[(i + stride) * inner_dim];
        sum += input_slice[(i + stride * 2) * inner_dim];
        sum += input_slice[(i + stride * 3) * inner_dim];
    }
    // Handle remainder
    for (; i < reduce_dim; i += stride) {
        sum += input_slice[i * inner_dim];
    }
    
    // Warp-level reduction
    sum = warp_reduce_sum(sum);
    
    // Shared memory for inter-warp reduction
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = BLOCK_SIZE / WARP_SIZE;
    
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        
        if (lane == 0) {
            output[batch_idx * inner_dim + inner_idx] = sum;
        }
    }
}

torch::Tensor sum_reduction_hip(torch::Tensor input, int dim) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D tensor");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Only float32 is supported");
    
    input = input.contiguous();
    
    const int batch_size = input.size(0);
    const int reduce_dim = input.size(1);
    const int inner_dim = input.size(2);
    
    auto output = torch::empty({batch_size, 1, inner_dim}, input.options());
    
    const int num_output_elements = batch_size * inner_dim;
    
    dim3 grid(num_output_elements);
    dim3 block(BLOCK_SIZE);
    
    sum_reduction_kernel_v2<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        reduce_dim,
        inner_dim
    );
    
    return output;
}
"""

sum_reduction_module = load_inline(
    name="sum_reduction",
    cpp_sources=sum_reduction_cpp_source,
    cuda_sources=sum_reduction_hip_source,
    functions=["sum_reduction_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "--offload-arch=gfx942"],
)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies sum reduction over the specified dimension using custom HIP kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        if self.dim == 1 and x.dim() == 3:
            return sum_reduction_module.sum_reduction_hip(x, self.dim)
        else:
            # Fallback to PyTorch for other cases
            return torch.sum(x, dim=self.dim, keepdim=True)


def custom_kernel(inputs):
    x = inputs[0]
    model = ModelNew(dim=1)
    return model(x)
