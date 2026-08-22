import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Ensure we build with HIPCC on ROCm
os.environ.setdefault("CXX", "hipcc")

bmm_rocblas_cpp_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <mutex>
#include <vector>

namespace {

rocblas_handle get_rocblas_handle_for_device(int device) {
    static std::mutex m;
    static std::vector<rocblas_handle> handles;

    std::lock_guard<std::mutex> g(m);
    if ((int)handles.size() <= device) {
        handles.resize(device + 1, nullptr);
    }
    if (handles[device] == nullptr) {
        // Ensure handle is created on the right device
        hipSetDevice(device);
        rocblas_handle h;
        rocblas_create_handle(&h);
        rocblas_set_pointer_mode(h, rocblas_pointer_mode_host);
        handles[device] = h;
    }
    return handles[device];
}

} // namespace

torch::Tensor bmm_rocblas(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA/HIP tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA/HIP tensor");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be FP32");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be FP32");
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3, "A and B must be 3D tensors");

    const auto batch = A.size(0);
    const auto m = A.size(1);
    const auto k = A.size(2);
    TORCH_CHECK(B.size(0) == batch && B.size(1) == k, "B shape mismatch");
    const auto n = B.size(2);

    at::cuda::CUDAGuard device_guard(A.device());

    auto A_c = A.contiguous();
    auto B_c = B.contiguous();
    auto C = torch::empty({batch, m, n}, A.options());

    // rocBLAS is column-major; use the standard row-major trick:
    // C_row(m,n) = A_row(m,k) * B_row(k,n)
    // Equivalent to computing D_col(n,m) = B_col(n,k) * A_col(k,m)
    // and storing into the same memory as C.

    float alpha = 1.0f;
    float beta = 0.0f;

    const int device = A.get_device();
    rocblas_handle handle = get_rocblas_handle_for_device(device);

    // Use the current PyTorch stream
    auto stream = at::cuda::getDefaultCUDAStream(device);
    rocblas_set_stream(handle, (hipStream_t)stream.stream());

    const rocblas_int M = (rocblas_int)n;
    const rocblas_int N = (rocblas_int)m;
    const rocblas_int K = (rocblas_int)k;

    const rocblas_int lda = M; // n
    const rocblas_int ldb = K; // k
    const rocblas_int ldc = M; // n

    const rocblas_stride strideA = (rocblas_stride)(n * k);
    const rocblas_stride strideB = (rocblas_stride)(k * m);
    const rocblas_stride strideC = (rocblas_stride)(n * m);

    rocblas_status status = rocblas_sgemm_strided_batched(
        handle,
        rocblas_operation_none,
        rocblas_operation_none,
        M, N, K,
        &alpha,
        (const float*)B_c.data_ptr<float>(), lda, strideA,
        (const float*)A_c.data_ptr<float>(), ldb, strideB,
        &beta,
        (float*)C.data_ptr<float>(), ldc, strideC,
        (rocblas_int)batch
    );

    TORCH_CHECK(status == rocblas_status_success, "rocblas_sgemm_strided_batched failed");
    return C;
}
"""

# Build extension once
_bmm_ext = load_inline(
    name="bmm_rocblas_ext",
    cpp_sources=bmm_rocblas_cpp_source,
    functions=["bmm_rocblas"],
    extra_cflags=["-O3"],
    extra_ldflags=["-lrocblas"],
    with_cuda=False,
    verbose=False,
)


class ModelNew(nn.Module):
    """Optimized batched matmul using a direct rocBLAS strided-batched SGEMM call."""

    def __init__(self):
        super().__init__()
        self._ext = _bmm_ext

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self._ext.bmm_rocblas(A, B)


# Keep the same input generators as the reference
batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

def get_inputs():
    A = torch.rand(batch_size, m, k)
    B = torch.rand(batch_size, k, n)
    return [A, B]

def get_init_inputs():
    return []
