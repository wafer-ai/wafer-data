
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

matmul_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <rocblas/rocblas.h>

static rocblas_handle global_handle = nullptr;

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    if (global_handle == nullptr) {
        rocblas_create_handle(&global_handle);
    }

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    float alpha = 1.0f;
    float beta = 0.0f;

    // rocblasSgemm expects column-major order.
    // PyTorch tensors are row-major.
    // For C = A * B (row-major):
    // C_row = A_row * B_row
    // Treat as C_col^T = B_row^T * A_row^T
    // Which is what we call in rocblas.
    
    rocblas_sgemm(global_handle,
                  rocblas_operation_none, rocblas_operation_none,
                  N, M, K,
                  &alpha,
                  B.data_ptr<float>(), N,
                  A.data_ptr<float>(), K,
                  &beta,
                  C.data_ptr<float>(), N);

    return C;
}
"""

matmul_module = load_inline(
    name="matmul_module",
    cpp_sources=matmul_cpp_source,
    functions=["matmul_hip"],
    verbose=True,
    extra_ldflags=["-lrocblas"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_module = matmul_module

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if not A.is_cuda:
            A = A.cuda()
        if not B.is_cuda:
            B = B.cuda()
        return self.matmul_module.matmul_hip(A, B)

def get_inputs():
    M, K, N = 8205, 2949, 5921
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []
