
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

argmax_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <limits>

__global__ void __launch_bounds__(256) argmax_kernel_general(const float* __restrict__ input, long* __restrict__ output,
                                     long outer_size, int reduce_size, long inner_size) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total_reductions = outer_size * inner_size;
    if (idx < total_reductions) {
        long outer_idx = idx / inner_size;
        long inner_idx = idx % inner_size;
        long base_idx = outer_idx * (long)reduce_size * inner_size + inner_idx;

        float max_val = -1e38f;
        int max_idx = 0;

        for (int i = 0; i < reduce_size; ++i) {
            float val = input[base_idx + (long)i * inner_size];
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }
        output[idx] = (long)max_idx;
    }
}

__global__ void __launch_bounds__(256) argmax_kernel_inner1(const float* __restrict__ input, long* __restrict__ output,
                                    long outer_size, int reduce_size) {
    long rid = blockIdx.x;
    if (rid < outer_size) {
        const float* row = input + rid * reduce_size;
        
        float max_val = -1e38f;
        int max_idx = 0;

        for (int i = threadIdx.x; i < reduce_size; i += blockDim.x) {
            float val = row[i];
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }

        __shared__ float shared_max[256];
        __shared__ int shared_idx[256];

        shared_max[threadIdx.x] = max_val;
        shared_idx[threadIdx.x] = max_idx;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                if (shared_max[threadIdx.x + s] > shared_max[threadIdx.x]) {
                    shared_max[threadIdx.x] = shared_max[threadIdx.x + s];
                    shared_idx[threadIdx.x] = shared_idx[threadIdx.x + s];
                } else if (shared_max[threadIdx.x + s] == shared_max[threadIdx.x]) {
                    if (shared_idx[threadIdx.x + s] < shared_idx[threadIdx.x]) {
                        shared_idx[threadIdx.x] = shared_idx[threadIdx.x + s];
                    }
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            output[rid] = (long)shared_idx[0];
        }
    }
}

torch::Tensor argmax_hip(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
    
    auto sizes = input.sizes();
    int64_t outer_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= sizes[i];
    int64_t reduce_size = sizes[dim];
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); ++i) inner_size *= sizes[i];

    auto output_sizes = sizes.vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options().dtype(torch::kLong));

    if (inner_size == 1) {
        const int block_size = 256;
        argmax_kernel_inner1<<<outer_size, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<long>(),
            outer_size, (int)reduce_size
        );
    } else {
        int64_t total_reductions = outer_size * inner_size;
        const int block_size = 256;
        const int64_t num_blocks = (total_reductions + block_size - 1) / block_size;
        argmax_kernel_general<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<long>(),
            outer_size, (int)reduce_size, inner_size
        );
    }

    return output;
}
"""

argmax_lib = load_inline(
    name="argmax_lib",
    cpp_sources=argmax_source,
    functions=["argmax_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return argmax_lib.argmax_hip(x, self.dim)

