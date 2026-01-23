
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

sum_reduction_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void sum_reduction_kernel(const float* __restrict__ x, float* __restrict__ out, 
                                    int pre, int mid, int post) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < pre && k < post) {
        float sum = 0.0f;
        int base_idx = (i * mid) * post + k;
        
        int j = 0;
        for (; j <= mid - 4; j += 4) {
            sum += x[base_idx + j * post];
            sum += x[base_idx + (j + 1) * post];
            sum += x[base_idx + (j + 2) * post];
            sum += x[base_idx + (j + 3) * post];
        }
        for (; j < mid; ++j) {
            sum += x[base_idx + j * post];
        }
        out[i * post + k] = sum;
    }
}

torch::Tensor sum_reduction_hip(torch::Tensor x, int dim) {
    auto shape = x.sizes();
    int pre = 1;
    for (int i = 0; i < dim; ++i) {
        pre *= shape[i];
    }
    int mid = shape[dim];
    int post = 1;
    for (int i = dim + 1; i < shape.size(); ++i) {
        post *= shape[i];
    }

    auto out_shape = shape.vec();
    out_shape[dim] = 1;
    auto out = torch::empty(out_shape, x.options());

    dim3 block_dim(256, 1);
    dim3 grid_dim((post + block_dim.x - 1) / block_dim.x, (pre + block_dim.y - 1) / block_dim.y);

    hipLaunchKernelGGL(sum_reduction_kernel, grid_dim, block_dim, 0, 0,
                       x.data_ptr<float>(), out.data_ptr<float>(), pre, mid, post);

    return out;
}
"""

sum_reduction_module = load_inline(
    name="sum_reduction",
    cpp_sources=sum_reduction_cpp_source,
    functions=["sum_reduction_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.sum_reduction = sum_reduction_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sum_reduction.sum_reduction_hip(x, self.dim)

