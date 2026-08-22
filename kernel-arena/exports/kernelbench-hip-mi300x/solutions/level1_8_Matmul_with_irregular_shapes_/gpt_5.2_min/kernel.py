import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Force HIP compiler
os.environ.setdefault("CXX", "hipcc")

# A thin wrapper over rocBLAS SGEMM. This should match or slightly improve PyTorch matmul
# for this specific 2D FP32 case by avoiding some dispatch overhead.

cpp_source = r"""
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <rocblas/rocblas.h>

// Cache a rocBLAS handle per device (simple; kernelbench uses one device)
static rocblas_handle g_handle = nullptr;

static inline rocblas_handle get_handle() {
    if (g_handle == nullptr) {
        rocblas_create_handle(&g_handle);
    }
    // Set stream to current PyTorch stream
    auto stream = at::hip::getDefaultHIPStream();
    rocblas_set_stream(g_handle, stream);
    return g_handle;
}

torch::Tensor matmul_rocblas_fp32(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be CUDA/HIP tensor");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA/HIP tensor");
    TORCH_CHECK(A.scalar_type() == at::kFloat, "A must be float32");
    TORCH_CHECK(B.scalar_type() == at::kFloat, "B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "K mismatch");

    // Ensure contiguous row-major
    if (!A.is_contiguous()) A = A.contiguous();
    if (!B.is_contiguous()) B = B.contiguous();

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    // rocBLAS assumes column-major by default. We compute C = A*B in row-major by using
    // (B^T * A^T)^T equivalence.
    // Column-major gemm: C_col(M,N) = op(A_col)*op(B_col)
    // Treat row-major A(M,K) as column-major A_col(K,M) with transpose.

    rocblas_handle handle = get_handle();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // We want C_row(M,N). In column-major, store as C_col(N,M) and transpose notion.
    // Compute: C_col(N,M) = B_col(N,K) * A_col(K,M)
    // where B_col is B in column-major if we interpret row-major B(K,N) as column-major B_col(N,K) without transpose.

    rocblas_operation transA = rocblas_operation_none; // B_col(N,K)
    rocblas_operation transB = rocblas_operation_none; // A_col(K,M)

    // Leading dimensions for column-major matrices
    const int64_t lda = N; // B_col has shape (N,K)
    const int64_t ldb = K; // A_col has shape (K,M)
    const int64_t ldc = N; // C_col has shape (N,M)

    // Pointer mapping:
    // B_row(K,N) data is same as B_col(N,K)
    // A_row(M,K) data is same as A_col(K,M)
    // C_row(M,N) data is same as C_col(N,M)

    // gemm: C_col(m,n) = A_col(m,k) * B_col(k,n)
    // We need C_col(N,M) = B_col(N,K) * A_col(K,M)
    // So set m=N, n=M, k=K, A=B_col, B=A_col

    rocblas_status st = rocblas_sgemm(handle,
                                     transA, transB,
                                     (rocblas_int)N, (rocblas_int)M, (rocblas_int)K,
                                     &alpha,
                                     (const float*)B.data_ptr<float>(), (rocblas_int)lda,
                                     (const float*)A.data_ptr<float>(), (rocblas_int)ldb,
                                     &beta,
                                     (float*)C.data_ptr<float>(), (rocblas_int)ldc);
    TORCH_CHECK(st == rocblas_status_success, "rocblas_sgemm failed");

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_rocblas_fp32", &matmul_rocblas_fp32, "MatMul via rocBLAS (FP32)");
}
"""

matmul_ext = load_inline(
    name="matmul_rocblas_fp32_ext",
    cpp_sources=cpp_source,
    functions=None,
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return matmul_ext.matmul_rocblas_fp32(A, B)
