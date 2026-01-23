
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

cpp_source = """
#include <torch/extension.h>
#include <rocblas/rocblas.h>

static rocblas_handle handle = nullptr;

torch::Tensor batched_gemm_hip(torch::Tensor A, torch::Tensor B) {
    if (handle == nullptr) {
        rocblas_create_handle(&handle);
    }

    const int batch_size = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int N = B.size(2);

    auto C = torch::empty({batch_size, M, N}, A.options());

    const float alpha = 1.0f;
    const float beta = 0.0f;

    long long int strideA = M * K;
    long long int strideB = K * N;
    long long int strideC = M * N;

    // A is (batch, M, K), B is (batch, K, N)
    // Row-major A*B = Column-major B^T * A^T
    // B^T is (N, K), A^T is (K, M) -> C^T is (N, M)
    
    rocblas_sgemm_strided_batched(
        handle,
        rocblas_operation_none,
        rocblas_operation_none,
        N, M, K,
        &alpha,
        B.data_ptr<float>(), N, strideB,
        A.data_ptr<float>(), K, strideA,
        &beta,
        C.data_ptr<float>(), N, strideC,
        batch_size
    );

    return C;
}
"""

batched_gemm = load_inline(
    name="batched_gemm",
    cpp_sources=cpp_source,
    functions=["batched_gemm_hip"],
    extra_cflags=["-I/opt/rocm/include"],
    extra_ldflags=["-L/opt/rocm/lib -lrocblas"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batched_gemm = batched_gemm

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.batched_gemm.batched_gemm_hip(A, B)

def get_inputs():
    batch_size = 128
    m = 128 * 4
    k = 256 * 4
    n = 512 * 4
    A = torch.rand(batch_size, m, k).cuda()
    B = torch.rand(batch_size, k, n).cuda()
    return [A, B]

def get_init_inputs():
    return []
