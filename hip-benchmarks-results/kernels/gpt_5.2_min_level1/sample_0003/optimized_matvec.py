import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ.setdefault("CXX", "hipcc")

# Use hipBLAS SGEMM with N=1. On some ROCm versions this can outperform SGEMV.
source = r'''
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>

#include <hipblas/hipblas.h>
#include <mutex>

static inline hipblasHandle_t get_handle() {
    static hipblasHandle_t handle = nullptr;
    static std::once_flag flag;
    std::call_once(flag, [](){
        hipblasCreate(&handle);
        hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
    });
    return handle;
}

torch::Tensor matvec_sgemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA/ROCm tensors");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32, "FP32 only");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "A and B must be contiguous");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A 2D, B 2D");

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    TORCH_CHECK(B.size(0) == K && B.size(1) == 1, "B must be (K,1)");

    auto C = torch::empty({M, 1}, A.options());

    // Column-major interpretation trick:
    // - Treat row-major A (M x K) as column-major A' (K x M) where A' = A^T.
    // - B is (K x 1); for N=1, row/col-major are identical.
    // Compute: C = op(A') * B with op(A') = transpose, giving (M x K) * (K x 1).

    const int m = (int)M;
    const int n = 1;
    const int k = (int)K;

    const int lda = (int)K; // A' is (k x m) col-major
    const int ldb = (int)K; // B is (k x 1)
    const int ldc = (int)M; // C is (m x 1)

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    hipblasHandle_t handle = get_handle();
    hipblasSetStream(handle, at::hip::getDefaultHIPStream());

    const float* A_ptr = (const float*)A.data_ptr<float>();
    const float* B_ptr = (const float*)B.data_ptr<float>();
    float* C_ptr = (float*)C.data_ptr<float>();

    hipblasStatus_t st = hipblasSgemm(handle,
                                     HIPBLAS_OP_T, HIPBLAS_OP_N,
                                     m, n, k,
                                     &alpha,
                                     A_ptr, lda,
                                     B_ptr, ldb,
                                     &beta,
                                     C_ptr, ldc);
    TORCH_CHECK(st == HIPBLAS_STATUS_SUCCESS, "hipblasSgemm failed");

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matvec_sgemm", &matvec_sgemm, "matvec via hipBLAS SGEMM (FP32, N=1)");
}
'''

ext = load_inline(
    name='matvec_sgemm_ext',
    cpp_sources='',
    cuda_sources=source,
    functions=None,
    extra_cuda_cflags=['-O3'],
    extra_cflags=['-O3'],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return ext.matvec_sgemm(A, B)


M = 256 * 8
K = 131072 * 8

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, 1)
    return [A, B]


def get_init_inputs():
    return []
