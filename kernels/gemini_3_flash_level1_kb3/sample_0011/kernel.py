
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

sum_reduction_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void sum_reduction_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                     int Pre, int Mid, int Post) {
    int total_output = Pre * Post;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_output) {
        int i = idx / Post;
        int k = idx % Post;
        float sum = 0.0f;
        
        // Use size_t for base_ptr to prevent overflow in index calculation
        const float* __restrict__ base_ptr = input + (size_t)i * Mid * Post + k;
        
        for (int j = 0; j < Mid; ++j) {
            sum += base_ptr[(size_t)j * Post];
        }
        output[idx] = sum;
    }
}

torch::Tensor sum_reduction_hip(torch::Tensor input, int dim) {
    auto shape = input.sizes().vec();
    int ndim = shape.size();
    if (dim < 0) dim += ndim;

    int Pre = 1;
    for (int i = 0; i < dim; ++i) Pre *= (int)shape[i];
    int Mid = (int)shape[dim];
    int Post = 1;
    for (int i = dim + 1; i < ndim; ++i) Post *= (int)shape[i];

    auto output_shape = shape;
    output_shape[dim] = 1;
    auto output = torch::empty(output_shape, input.options());

    int total_output = Pre * Post;
    const int block_size = 256;
    const int num_blocks = (total_output + block_size - 1) / block_size;

    sum_reduction_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        Pre, Mid, Post
    );

    return output;
}
"""

sum_reduction_lib = load_inline(
    name="sum_reduction_lib",
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

def get_inputs():
    batch_size = 128
    dim1 = 4096
    dim2 = 4095
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    reduce_dim = 1
    return [reduce_dim]
