import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Use rocBLAS for optimized GEMM
matmul_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <rocblas/rocblas.h>

// Global rocBLAS handle
static rocblas_handle handle = nullptr;
static bool initialized = false;

void init_rocblas() {
    if (!initialized) {
        rocblas_create_handle(&handle);
        initialized = true;
    }
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    init_rocblas();
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // rocBLAS uses column-major, but our tensors are row-major
    // C = A * B in row-major is equivalent to C^T = B^T * A^T in column-major
    // So we call: C^T (N x M) = B^T (N x K) * A^T (K x M)
    rocblas_sgemm(
        handle,
        rocblas_operation_none,  // B is used as B^T (column-major view of row-major B)
        rocblas_operation_none,  // A is used as A^T (column-major view of row-major A)
        N,  // rows of op(B) = N
        M,  // cols of op(A) = M  
        K,  // inner dimension
        &alpha,
        B.data_ptr<float>(),  // B^T in column major = B in row major
        N,  // leading dimension of B (row-major stride)
        A.data_ptr<float>(),  // A^T in column major = A in row major
        K,  // leading dimension of A (row-major stride)
        &beta,
        C.data_ptr<float>(),
        N   // leading dimension of C (row-major stride)
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
    extra_ldflags=["-L/opt/rocm/lib", "-lrocblas"]
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_module
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.matmul_hip(A, B)


def get_inputs():
    M = 1024 * 2
    K = 4096 * 2
    N = 2048 * 2
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]


def get_init_inputs():
    return []
