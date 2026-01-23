
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

mean_reduction_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void mean_reduction_element_kernel(const float* x, float* out, int64_t outer, int64_t reduction, int64_t inner) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_output_elements = outer * inner;
    if (idx < total_output_elements) {
        int64_t outer_idx = idx / inner;
        int64_t inner_idx = idx % inner;
        float sum = 0;
        const float* base_ptr = x + outer_idx * reduction * inner + inner_idx;
        
        int64_t k = 0;
        for (; k <= reduction - 8; k += 8) {
            sum += base_ptr[k * inner];
            sum += base_ptr[(k + 1) * inner];
            sum += base_ptr[(k + 2) * inner];
            sum += base_ptr[(k + 3) * inner];
            sum += base_ptr[(k + 4) * inner];
            sum += base_ptr[(k + 5) * inner];
            sum += base_ptr[(k + 6) * inner];
            sum += base_ptr[(k + 7) * inner];
        }
        for (; k < reduction; k++) {
            sum += base_ptr[k * inner];
        }
        out[idx] = sum / (float)reduction;
    }
}

template <int BLOCK_SIZE>
__global__ void mean_reduction_block_kernel(const float* x, float* out, int64_t outer, int64_t reduction, int64_t inner) {
    int64_t out_idx = blockIdx.x;
    int64_t outer_idx = out_idx / inner;
    int64_t inner_idx = out_idx % inner;
    int tid = threadIdx.x;

    float sum = 0;
    const float* base_ptr = x + outer_idx * reduction * inner + inner_idx;
    for (int64_t k = tid; k < reduction; k += BLOCK_SIZE) {
        sum += base_ptr[k * inner];
    }

    __shared__ float shared_sum[BLOCK_SIZE];
    shared_sum[tid] = sum;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[out_idx] = shared_sum[0] / (float)reduction;
    }
}

torch::Tensor mean_reduction_hip(torch::Tensor x, int64_t dim) {
    if (dim < 0) dim += x.dim();
    x = x.contiguous();
    
    auto shape = x.sizes();
    int64_t outer = 1;
    for (int i = 0; i < dim; i++) outer *= shape[i];
    int64_t reduction = shape[dim];
    int64_t inner = 1;
    for (int i = dim + 1; i < x.dim(); i++) inner *= shape[i];

    auto out_shape = shape.vec();
    out_shape.erase(out_shape.begin() + dim);
    auto out = torch::empty(out_shape, x.options());

    int64_t total_output_elements = outer * inner;
    
    if (inner >= 64) {
        const int block_size = 256;
        const int64_t num_blocks = (total_output_elements + block_size - 1) / block_size;
        mean_reduction_element_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), outer, reduction, inner);
    } else {
        const int block_size = 256;
        mean_reduction_block_kernel<block_size><<<total_output_elements, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), outer, reduction, inner);
    }

    return out;
}
"""

mean_reduction_lib = load_inline(
    name="mean_reduction_lib_v3",
    cpp_sources=mean_reduction_source,
    functions=["mean_reduction_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mean_reduction_lib.mean_reduction_hip(x, self.dim)

def get_inputs():
    batch_size = 128
    dim1 = 4096
    dim2 = 4095
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [1]
