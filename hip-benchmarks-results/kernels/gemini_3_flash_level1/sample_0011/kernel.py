
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

sum_reduction_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void __launch_bounds__(256)
sum_reduction_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                    size_t M, size_t D, size_t N) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = M * N;

    if (idx < total_elements) {
        size_t m = idx / N;
        size_t n = idx % N;
        
        float sum = 0.0f;
        size_t base_idx = m * D * N + n;
        
        size_t i = 0;
        for (; i + 15 < D; i += 16) {
            sum += input[base_idx + i * N];
            sum += input[base_idx + (i + 1) * N];
            sum += input[base_idx + (i + 2) * N];
            sum += input[base_idx + (i + 3) * N];
            sum += input[base_idx + (i + 4) * N];
            sum += input[base_idx + (i + 5) * N];
            sum += input[base_idx + (i + 6) * N];
            sum += input[base_idx + (i + 7) * N];
            sum += input[base_idx + (i + 8) * N];
            sum += input[base_idx + (i + 9) * N];
            sum += input[base_idx + (i + 10) * N];
            sum += input[base_idx + (i + 11) * N];
            sum += input[base_idx + (i + 12) * N];
            sum += input[base_idx + (i + 13) * N];
            sum += input[base_idx + (i + 14) * N];
            sum += input[base_idx + (i + 15) * N];
        }
        for (; i < D; ++i) {
            sum += input[base_idx + i * N];
        }
        output[idx] = sum;
    }
}

torch::Tensor sum_reduction_hip(torch::Tensor input, int dim) {
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }
    
    auto sizes = input.sizes();
    int ndim = input.dim();
    
    if (dim < 0) dim += ndim;

    size_t M = 1;
    for (int i = 0; i < dim; ++i) {
        M *= sizes[i];
    }
    
    size_t D = sizes[dim];
    
    size_t N = 1;
    for (int i = dim + 1; i < ndim; ++i) {
        N *= sizes[i];
    }

    auto output_shape = sizes.vec();
    output_shape[dim] = 1;
    auto output = torch::empty(output_shape, input.options());

    size_t total_output_elements = M * N;
    const int block_size = 256;
    const int num_blocks = (total_output_elements + block_size - 1) / block_size;

    sum_reduction_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        M, D, N
    );

    return output;
}
"""

sum_reduction_lib = load_inline(
    name="sum_reduction_lib_v3",
    cpp_sources=sum_reduction_source,
    functions=["sum_reduction_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sum_reduction_lib.sum_reduction_hip(x, self.dim)
