import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Use rocBLAS directly for optimal performance
bmm_hip_source = """
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <torch/extension.h>

// Global hipBLAS handle
hipblasHandle_t handle = nullptr;

void init_hipblas() {
    if (handle == nullptr) {
        hipblasCreate(&handle);
    }
}

torch::Tensor batched_matmul_hip(torch::Tensor A, torch::Tensor B) {
    init_hipblas();
    
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions must match");
    TORCH_CHECK(A.is_cuda(), "A must be on CUDA");
    TORCH_CHECK(B.is_cuda(), "B must be on CUDA");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    
    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);
    
    auto C = torch::empty({batch_size, M, N}, A.options());
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // hipBLAS uses column-major, so we compute C^T = B^T * A^T
    // which gives C = A * B in row-major
    // For row-major: C[m,n] = sum_k A[m,k] * B[k,n]
    // In col-major with transposition:
    // C = A * B becomes C^T = B^T * A^T
    
    // Strides for batched operation
    long long strideA = M * K;
    long long strideB = K * N;
    long long strideC = M * N;
    
    // hipblasSgemmStridedBatched parameters for row-major matrices:
    // We want C[b][i][j] = sum_k A[b][i][k] * B[b][k][j]
    // Using the formula: to compute C = A * B in row-major,
    // call gemm with C^T = B^T * A^T in column-major
    // i.e., gemm(N, N, M, N, K, alpha, B, N, strideB, A, K, strideA, beta, C, N, strideC)
    
    hipblasStatus_t status = hipblasSgemmStridedBatched(
        handle,
        HIPBLAS_OP_N,  // B is not transposed (as B^T in col-major = B in row-major)
        HIPBLAS_OP_N,  // A is not transposed (as A^T in col-major = A in row-major)
        N,             // number of rows of C (col-major) = number of cols (row-major)
        M,             // number of cols of C (col-major) = number of rows (row-major)
        K,             // inner dimension
        &alpha,
        B.data_ptr<float>(), N, strideB,  // B: K x N in row-major
        A.data_ptr<float>(), K, strideA,  // A: M x K in row-major
        &beta,
        C.data_ptr<float>(), N, strideC,  // C: M x N in row-major
        batch_size
    );
    
    TORCH_CHECK(status == HIPBLAS_STATUS_SUCCESS, "hipBLAS SGEMM failed");
    
    return C;
}
"""

bmm_cpp_source = """
torch::Tensor batched_matmul_hip(torch::Tensor A, torch::Tensor B);
"""

bmm_module = load_inline(
    name="batched_matmul_hipblas",
    cpp_sources=bmm_cpp_source,
    cuda_sources=bmm_hip_source,
    functions=["batched_matmul_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
    extra_ldflags=["-lhipblas"]
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.bmm_op = bmm_module
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.bmm_op.batched_matmul_hip(A, B)


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
