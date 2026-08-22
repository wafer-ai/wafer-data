
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

fused_softmax_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

__global__ void fast_softmax_kernel(float* x, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    float local_max = -1e38f;
    for (int col = tid; col < cols; col += num_threads) {
        local_max = fmaxf(local_max, x[row * cols + col]);
    }

    __shared__ float s_reduce[32];
    for (int offset = 32; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, offset));
    if (tid % 32 == 0) s_reduce[tid / 32] = local_max;
    __syncthreads();
    if (tid < 32) {
        float b_max = (tid < num_threads / 32) ? s_reduce[tid] : -1e38f;
        for (int offset = 16; offset > 0; offset >>= 1)
            b_max = fmaxf(b_max, __shfl_xor(b_max, offset));
        s_reduce[0] = b_max;
    }
    __syncthreads();
    float max_val = s_reduce[0];

    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += num_threads) {
        float val = expf(x[row * cols + col] - max_val);
        x[row * cols + col] = val;
        local_sum += val;
    }

    for (int offset = 32; offset > 0; offset >>= 1)
        local_sum += __shfl_xor(local_sum, offset);
    if (tid % 32 == 0) s_reduce[tid / 32] = local_sum;
    __syncthreads();
    if (tid < 32) {
        float b_sum = (tid < num_threads / 32) ? s_reduce[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            b_sum += __shfl_xor(b_sum, offset);
        s_reduce[0] = b_sum;
    }
    __syncthreads();
    float sum_val = s_reduce[0];

    for (int col = tid; col < cols; col += num_threads) {
        x[row * cols + col] /= sum_val;
    }
}

torch::Tensor fast_softmax_hip(torch::Tensor x) {
    int rows = x.size(0);
    int cols = x.size(1);
    fast_softmax_kernel<<<rows, 1024>>>(x.data_ptr<float>(), rows, cols);
    return x;
}
"""

softmax_lib = load_inline(
    name="softmax_lib",
    cpp_sources=fused_softmax_source,
    functions=["fast_softmax_hip"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # Using addmm to keep the GEMM part as fast as possible.
        x = torch.addmm(self.matmul.bias, x, self.matmul.weight.t())
        x = self.dropout(x)
        return softmax_lib.fast_softmax_hip(x)

def get_inputs():
    return [torch.rand(128, 16384).cuda()]

def get_init_inputs():
    return [16384, 16384, 0.2]
