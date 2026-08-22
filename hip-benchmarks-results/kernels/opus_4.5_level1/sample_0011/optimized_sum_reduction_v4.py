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

// Block reduction kernel for contiguous data
// After transposition: shape is (batch_size, inner_dim, reduce_dim)
// Reduces last dimension

#define BLOCK_SIZE 256
#define WARP_SIZE 64

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__global__ void sum_reduction_contiguous_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int num_outputs,
    const int reduce_dim
) {
    int out_idx = blockIdx.x;
    if (out_idx >= num_outputs) return;
    
    const float* input_row = input + out_idx * reduce_dim;
    
    // Each thread accumulates partial sum
    float sum = 0.0f;
    
    for (int i = threadIdx.x; i < reduce_dim; i += blockDim.x) {
        sum += input_row[i];
    }
    
    // Warp-level reduction
    sum = warp_reduce_sum(sum);
    
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = BLOCK_SIZE / WARP_SIZE;
    
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    
    __syncthreads();
    
    if (warp_id == 0) {
        sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        
        if (lane == 0) {
            output[out_idx] = sum;
        }
    }
}

// Non-contiguous reduction with strided access
__global__ void sum_reduction_strided_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int reduce_dim,
    const int inner_dim
) {
    int out_idx = blockIdx.x;
    int batch_idx = out_idx / inner_dim;
    int inner_idx = out_idx % inner_dim;
    
    if (batch_idx >= batch_size) return;
    
    const float* input_slice = input + batch_idx * reduce_dim * inner_dim + inner_idx;
    
    float sum = 0.0f;
    
    for (int i = threadIdx.x; i < reduce_dim; i += blockDim.x) {
        sum += input_slice[i * inner_dim];
    }
    
    sum = warp_reduce_sum(sum);
    
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = BLOCK_SIZE / WARP_SIZE;
    
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    
    __syncthreads();
    
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
    
    sum_reduction_strided_kernel<<<grid, block>>>(
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
            return torch.sum(x, dim=self.dim, keepdim=True)


def custom_kernel(inputs):
    x = inputs[0]
    model = ModelNew(dim=1)
    return model(x)
