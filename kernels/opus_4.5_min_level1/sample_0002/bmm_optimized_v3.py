import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

bmm_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <rocblas/rocblas.h>

// Global rocBLAS handle (initialized once)
static rocblas_handle handle = nullptr;
static bool handle_initialized = false;

void init_rocblas_handle() {
    if (!handle_initialized) {
        rocblas_create_handle(&handle);
        handle_initialized = true;
    }
}

torch::Tensor batched_matmul_hip(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions must match");
    
    init_rocblas_handle();
    
    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);
    
    auto C = torch::empty({batch_size, M, N}, A.options());
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // rocBLAS uses column-major, but PyTorch uses row-major
    // C = A * B in row-major is equivalent to C^T = B^T * A^T in column-major
    // So we compute C^T = B^T * A^T, which gives us C in row-major format
    
    // Get strides for strided batched GEMM
    long long int strideA = M * K;
    long long int strideB = K * N;
    long long int strideC = M * N;
    
    // In row-major: C[m,n] = sum_k A[m,k] * B[k,n]
    // Treating as column-major with transposition:
    // rocBLAS sees matrices as column-major, so:
    // A (M x K) row-major = A^T (K x M) column-major
    // B (K x N) row-major = B^T (N x K) column-major
    // C (M x N) row-major = C^T (N x M) column-major
    // We want C = A * B in row-major
    // In column-major: C^T = B^T * A^T
    // So we call gemm with:
    //   op(B^T) = B^T (no transpose, since B is already row-major = B^T column-major)
    //   op(A^T) = A^T (no transpose, since A is already row-major = A^T column-major)
    //   C^T = op(B^T) * op(A^T) = B^T * A^T
    // Dimensions for column-major:
    //   B^T is (N x K), A^T is (K x M), C^T is (N x M)
    //   So we do gemm(N, M, K, B, A, C)
    
    rocblas_sgemm_strided_batched(
        handle,
        rocblas_operation_none,  // op(B)
        rocblas_operation_none,  // op(A)
        N,                       // m (rows of B^T and C^T)
        M,                       // n (cols of A^T and C^T)
        K,                       // k
        &alpha,
        B.data_ptr<float>(),     // B^T in column-major = B in row-major
        N,                       // lda (leading dimension of B^T = N)
        strideB,
        A.data_ptr<float>(),     // A^T in column-major = A in row-major  
        K,                       // ldb (leading dimension of A^T = K)
        strideA,
        &beta,
        C.data_ptr<float>(),
        N,                       // ldc (leading dimension of C^T = N)
        strideC,
        batch_size
    );
    
    return C;
}
"""

bmm_cpp_source = """
torch::Tensor batched_matmul_hip(torch::Tensor A, torch::Tensor B);
"""

bmm_module = load_inline(
    name="bmm_hip_v3",
    cpp_sources=bmm_cpp_source,
    cuda_sources=bmm_hip_source,
    functions=["batched_matmul_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"],
    extra_ldflags=["-lrocblas"]
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.bmm_op = bmm_module
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.bmm_op.batched_matmul_hip(A.contiguous(), B.contiguous())


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
