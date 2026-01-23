
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Set HIP compiler
os.environ["CXX"] = "hipcc"

# HIP kernel for matrix-scalar multiplication
matrix_scalar_mult_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void matrix_scalar_mult_kernel_float4(const float4* A, float s, float4* C, int64_t n_vec) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec) {
        float4 val = A[idx];
        val.x *= s;
        val.y *= s;
        val.z *= s;
        val.w *= s;
        C[idx] = val;
    }
}

torch::Tensor matrix_scalar_mult_hip(torch::Tensor A, float s) {
    auto n = A.numel();
    auto C = torch::empty_like(A);
    
    if (n % 4 == 0) {
        int64_t n_vec = n / 4;
        const int block_size = 256;
        const int64_t num_blocks = (n_vec + block_size - 1) / block_size;
        matrix_scalar_mult_kernel_float4<<<num_blocks, block_size>>>(
            reinterpret_cast<const float4*>(A.data_ptr<float>()),
            s,
            reinterpret_cast<float4*>(C.data_ptr<float>()),
            n_vec
        );
    } else {
        at::mul_out(C, A, s);
    }
    
    return C;
}
"""

matrix_scalar_mult_module = load_inline(
    name="matrix_scalar_mult",
    cpp_sources=matrix_scalar_mult_source,
    functions=["matrix_scalar_mult_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.module = matrix_scalar_mult_module

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return self.module.matrix_scalar_mult_hip(A, s)

M = 16384 * 4
N = 4096 * 4

def get_inputs():
    A = torch.rand(M, N).cuda()
    s = 3.14
    return [A, s]

def get_init_inputs():
    return []
