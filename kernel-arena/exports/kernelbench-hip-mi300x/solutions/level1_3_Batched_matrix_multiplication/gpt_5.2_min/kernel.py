import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Ensure we build with hipcc on ROCm
os.environ.setdefault("CXX", "hipcc")

src = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <rocblas/rocblas.h>

// Simple rocBLAS-backed strided batched SGEMM for PyTorch row-major BMM.
// Computes: C[b] = A[b] (m x k) * B[b] (k x n), FP32.
// Mapping to rocBLAS (column-major):
//   Treat B (k x n) row-major as column-major (n x k) = B^T.
//   Treat A (m x k) row-major as column-major (k x m) = A^T.
// Then compute column-major: (n x m) = (n x k) * (k x m) => C^T.
// The (n x m) column-major layout matches (m x n) row-major layout.

torch::Tensor bmm_rocblas_fp32(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA/HIP tensors");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32,
                "Only FP32 supported");
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3, "A and B must be 3D");

    // Enforce contiguous for predictable strides
    A = A.contiguous();
    B = B.contiguous();

    const auto batch = (int)A.size(0);
    const auto m = (int)A.size(1);
    const auto k = (int)A.size(2);
    TORCH_CHECK(B.size(0) == batch && B.size(1) == k, "B shape mismatch");
    const auto n = (int)B.size(2);

    auto C = torch::empty({batch, m, n}, A.options());

    // rocBLAS handle (static to avoid per-call create/destroy overhead)
    static rocblas_handle handle = nullptr;
    static std::once_flag once;
    std::call_once(once, [](){
        rocblas_create_handle(&handle);
        // We control pointer mode per call.
    });

    // Use the current PyTorch stream
    auto stream = at::cuda::getDefaultCUDAStream();
    rocblas_set_stream(handle, (hipStream_t)stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // rocBLAS column-major GEMM sizes
    // m_rocblas = n, n_rocblas = m, k_rocblas = k
    const int m_rb = n;
    const int n_rb = m;
    const int k_rb = k;

    // A_rb points to B tensor data, interpreted as column-major (n x k)
    const float* A_rb = (const float*)B.data_ptr<float>();
    // B_rb points to A tensor data, interpreted as column-major (k x m)
    const float* B_rb = (const float*)A.data_ptr<float>();
    float* C_rb = (float*)C.data_ptr<float>();

    const int lda = m_rb; // n
    const int ldb = k_rb; // k
    const int ldc = m_rb; // n

    const long long strideA = (long long)n * (long long)k; // per-batch elements of B
    const long long strideB = (long long)k * (long long)m; // per-batch elements of A
    const long long strideC = (long long)n * (long long)m;

    rocblas_status st = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    TORCH_CHECK(st == rocblas_status_success, "rocblas_set_pointer_mode failed");

    st = rocblas_sgemm_strided_batched(
        handle,
        rocblas_operation_none,
        rocblas_operation_none,
        m_rb, n_rb, k_rb,
        &alpha,
        A_rb, lda, strideA,
        B_rb, ldb, strideB,
        &beta,
        C_rb, ldc, strideC,
        batch
    );

    TORCH_CHECK(st == rocblas_status_success, "rocblas_sgemm_strided_batched failed");
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bmm_rocblas_fp32", &bmm_rocblas_fp32, "BMM via rocBLAS (FP32)");
}
'''

bmm_ext = load_inline(
    name="bmm_rocblas_ext",
    cpp_sources=src,
    functions=None,
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = bmm_ext

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.ext.bmm_rocblas_fp32(A, B)


def get_inputs():
    batch_size = 128
    m = 128 * 4
    k = 256 * 4
    n = 512 * 4
    A = torch.rand(batch_size, m, k, device="cuda", dtype=torch.float32)
    B = torch.rand(batch_size, k, n, device="cuda", dtype=torch.float32)
    return [A, B]


def get_init_inputs():
    return []
