import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

M = 8205
K = 2949
N = 5921

os.environ["CXX"] = "hipcc"

matmul_cpp_source = r'''
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void sgemm_tiled(const float *A, const float *B, float *C, int M, int N, int K) {
  constexpr int TS = 32;
  __shared__ float sh_As[2][TS][TS];
  __shared__ float sh_Bs[2][TS][TS];

  const int by = blockIdx.y;
  const int bx = blockIdx.x;
  const int ty = threadIdx.y;
  const int tx = threadIdx.x;

  float acc = 0.0f;
  const int ktile_max = (K + TS - 1) / TS;
  int buf_idx = 0;

  // Preload first tile kz = 0
  {
    const int kz_load = 0;
    int row_a = by * TS + ty;
    int col_a = kz_load * TS + tx;
    sh_As[buf_idx][ty][tx] = (row_a < M && col_a < K) ? A[row_a * K + col_a] : 0.0f;

    int row_b = kz_load * TS + ty;
    int col_b = bx * TS + tx;
    sh_Bs[buf_idx][ty][tx] = (row_b < K && col_b < N) ? B[row_b * N + col_b] : 0.0f;
  }
  __syncthreads();

  // Loop over k-tiles
  for (int kz = 0; kz < ktile_max; ++kz) {
    // Compute from current buffer
    #pragma unroll
    for (int kk = 0; kk < TS; ++kk) {
      acc += sh_As[buf_idx][ty][kk] * sh_Bs[buf_idx][kk][tx];
    }

    // Switch buffer
    buf_idx = 1 - buf_idx;

    // Load next tile if any
    if (kz + 1 < ktile_max) {
      const int kz_load = kz + 1;
      int row_a = by * TS + ty;
      int col_a = kz_load * TS + tx;
      sh_As[buf_idx][ty][tx] = (row_a < M && col_a < K) ? A[row_a * K + col_a] : 0.0f;

      int row_b = kz_load * TS + ty;
      int col_b = bx * TS + tx;
      sh_Bs[buf_idx][ty][tx] = (row_b < K && col_b < N) ? B[row_b * N + col_b] : 0.0f;
      __syncthreads();
    }
  }

  // Store result
  int row_c = by * TS + ty;
  int col_c = bx * TS + tx;
  if (row_c < M && col_c < N) {
    C[row_c * N + col_c] = acc;
  }
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
  int64_t m = A.size(0);
  int64_t k = A.size(1);
  int64_t n = B.size(1);
  TORCH_CHECK(k == B.size(0), "matmul: input shapes don't match");

  auto options = A.options();
  torch::Tensor C = torch::zeros({m, n}, options);

  constexpr int TS = 32;
  dim3 block(TS, TS);
  dim3 grid((n + TS - 1) / TS, (m + TS - 1) / TS);

  sgemm_tiled<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), (int)m, (int)n, (int)k);

  return C;
}
'''

matmul_mod = load_inline(
    name="matmul",
    cpp_sources=[matmul_cpp_source],
    functions=["matmul_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_hip = matmul_mod.matmul_hip

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.contiguous().to("cuda")
        B = B.contiguous().to("cuda")
        return self.matmul_hip(A, B)

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]

def get_init_inputs():
    return []
