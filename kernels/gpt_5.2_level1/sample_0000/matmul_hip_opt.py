import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Force HIP compilation path on ROCm
os.environ.setdefault("CXX", "hipcc")

# A thin wrapper around ATen's BLAS GEMM for FP32.
# For row-major C = A @ B, we call column-major GEMM on swapped operands so the
# raw memory layout matches without an explicit transpose.

matmul_cpp = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDABlas.h>

static inline void check_inputs(const torch::Tensor& A, const torch::Tensor& B) {
  TORCH_CHECK(A.is_cuda(), "A must be a CUDA/HIP tensor");
  TORCH_CHECK(B.is_cuda(), "B must be a CUDA/HIP tensor");
  TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
  TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
  TORCH_CHECK(A.size(1) == B.size(0), "Incompatible shapes for matmul");
}

torch::Tensor matmul_sgemm_hip(torch::Tensor A, torch::Tensor B) {
  check_inputs(A, B);

  // Benchmark inputs are contiguous, but keep correctness for general callers.
  auto A_ = A.contiguous();
  auto B_ = B.contiguous();

  const auto m = A_.size(0);
  const auto k = A_.size(1);
  const auto n = B_.size(1);

  auto C = torch::empty({m, n}, A_.options());

  at::cuda::CUDAGuard device_guard(A_.device());

  const float alpha = 1.0f;
  const float beta = 0.0f;

  // at::cuda::blas::gemm is column-major.
  // Row-major C(m,n) = A(m,k) * B(k,n)
  // is equivalent in raw memory to column-major C_col(n,m) = B_col(n,k) * A_col(k,m)
  // when interpreting row-major matrices as transposed column-major.
  at::cuda::blas::gemm<float>(
      'n', 'n',
      n, m, k,
      alpha,
      B_.data_ptr<float>(), n,
      A_.data_ptr<float>(), k,
      beta,
      C.data_ptr<float>(), n);

  return C;
}
'''

matmul_ext = load_inline(
    name="matmul_sgemm_hip_ext",
    cpp_sources=matmul_cpp,
    functions=["matmul_sgemm_hip"],
    extra_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.ext = matmul_ext

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.ext.matmul_sgemm_hip(A, B)


# Reference-style helpers
N = 2048 * 2

def get_inputs():
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    return [A, B]


def get_init_inputs():
    return []
