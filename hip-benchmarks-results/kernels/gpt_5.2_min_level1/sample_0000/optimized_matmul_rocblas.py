import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Force HIP compilation
os.environ.setdefault("CXX", "hipcc")

# rocBLAS-backed SGEMM wrapper.
# We intentionally use rocBLAS' column-major GEMM and exploit the row-major/column-major
# interpretation trick to get correct row-major output without an explicit transpose:
# rocBLAS computes C_col = B_col * A_col = (B_row^T) * (A_row^T) = (A_row * B_row)^T.
# PyTorch interprets the output memory as row-major, i.e. C_row = C_col^T = A_row * B_row.

cpp_src = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

namespace {

static rocblas_handle get_handle() {
    // One handle per host thread; reused across calls.
    thread_local rocblas_handle handle = nullptr;
    if (!handle) {
        rocblas_create_handle(&handle);
        // Prefer host pointer mode for alpha/beta on host.
        rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    }
    return handle;
}

} // anonymous namespace

torch::Tensor matmul_rocblas_sgemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA/HIP tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "FP32 only");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "2D matrices only");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have same shape");

    // Make contiguous to satisfy leading-dimension assumptions.
    A = A.contiguous();
    B = B.contiguous();

    const int64_t N64 = A.size(0);
    TORCH_CHECK(N64 <= INT_MAX, "N too large");
    const int N = (int)N64;

    auto C = torch::empty({N, N}, A.options());

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    rocblas_handle handle = get_handle();
    hipStream_t stream = at::hip::getDefaultHIPStream();
    rocblas_set_stream(handle, stream);

    // Column-major SGEMM: C = alpha * op(A) * op(B) + beta * C.
    // To produce correct row-major C for PyTorch, compute (A_row * B_row)^T via:
    //   C_col = B_col * A_col (no transposes, swapped inputs)
    // where A_col == A_row^T and B_col == B_row^T due to memory layout.
    rocblas_status st = rocblas_sgemm(
        handle,
        rocblas_operation_none,
        rocblas_operation_none,
        N, N, N,
        &alpha,
        (const float*)B.data_ptr<float>(), N,
        (const float*)A.data_ptr<float>(), N,
        &beta,
        (float*)C.data_ptr<float>(), N
    );
    TORCH_CHECK(st == rocblas_status_success, "rocblas_sgemm failed with status ", (int)st);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_rocblas_sgemm", &matmul_rocblas_sgemm, "rocBLAS SGEMM (FP32)");
}
"""

ext = load_inline(
    name="matmul_rocblas_ext",
    cpp_sources=cpp_src,
    functions=None,
    extra_cuda_cflags=["-O3"],
    extra_ldflags=["-lrocblas"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = ext

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.ext.matmul_rocblas_sgemm(A, B)
