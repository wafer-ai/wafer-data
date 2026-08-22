import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Use rocBLAS for high-performance matrix multiplication
matmul_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <rocblas/rocblas.h>

static rocblas_handle handle = nullptr;

void init_rocblas() {
    if (handle == nullptr) {
        rocblas_create_handle(&handle);
    }
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    init_rocblas();
    
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::empty({M, N}, A.options());
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // rocBLAS uses column-major ordering, so we compute B^T * A^T = (A * B)^T
    // C = A * B in row-major is equivalent to C^T = B^T * A^T in column-major
    // Since we're using row-major, we swap A and B and transpose the dimensions
    rocblas_sgemm(
        handle,
        rocblas_operation_none,  // op(B) = B
        rocblas_operation_none,  // op(A) = A
        N,  // rows of op(B) and C
        M,  // cols of op(A) and C  
        K,  // cols of op(B) = rows of op(A)
        &alpha,
        B.data_ptr<float>(), N,  // B is K x N in row-major, so ldb = N
        A.data_ptr<float>(), K,  // A is M x K in row-major, so lda = K
        &beta,
        C.data_ptr<float>(), N   // C is M x N in row-major, so ldc = N
    );
    
    return C;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B);
"""

matmul_module = load_inline(
    name="matmul_hip",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_hip_source,
    functions=["matmul_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
    extra_ldflags=["-lrocblas"]
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_module

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.matmul_hip(A, B)


def get_inputs():
    M = 8205
    K = 2949
    N = 5921
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]


def get_init_inputs():
    return []
