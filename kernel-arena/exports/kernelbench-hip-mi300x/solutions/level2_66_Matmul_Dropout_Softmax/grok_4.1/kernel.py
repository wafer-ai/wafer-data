import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

softmax_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void softmax_kernel(const float *input, float *output, int rows, int cols) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    const float *row_input = input + row * cols;
    float *row_output = output + row * cols;
    int seg_size = (cols + block_size - 1) / block_size;
    int start = tid * seg_size;
    int end = start + seg_size;
    if (end > cols) end = cols;
    float max_val = -3.4e+38f;
    for (int i = start; i < end; ++i) {
        max_val = fmaxf(max_val, row_input[i]);
    }
    shared[tid] = max_val;
    __syncthreads();
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }
    float global_max = shared[0];
    float sum_exp = 0.0f;
    for (int i = start; i < end; ++i) {
        sum_exp += expf(row_input[i] - global_max);
    }
    shared[tid] = sum_exp;
    __syncthreads();
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    float global_sum = shared[0];
    if (global_sum == 0.0f) global_sum = 1e-10f;
    for (int i = start; i < end; ++i) {
        row_output[i] = expf(row_input[i] - global_max) / global_sum;
    }
}

torch::Tensor softmax_hip(torch::Tensor input) {
    auto sizes = input.sizes();
    int rows = sizes[0];
    int cols = sizes[1];
    auto output = torch::empty_like(input);
    const int block_size = 1024;
    dim3 block(block_size);
    dim3 grid(rows);
    size_t shmem_size = block_size * sizeof(float);
    hipLaunchKernelGGL(softmax_kernel, grid, block, shmem_size, 0, input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    return output;
}
"""

softmax_module = load_inline(
    name="softmax_hip",
    cpp_sources=softmax_cpp_source,
    functions=["softmax_hip"],
    verbose=True,
)

batch_size = 128
in_features = 16384
out_features = 16384
dropout_p = 0.2

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.softmax_module = softmax_module

    def forward(self, x):
        x = self.linear(x)
        x = self.softmax_module.softmax_hip(x)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, dropout_p]
