import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void gemm_tiled_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
  __shared__ float Ashmem[16][16];
  __shared__ float Bshmem[16][16];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  float sum = 0.0f;

  for (int kid = 0; kid < K; kid += 16) {
    // Load A tile
    if (by * 16 + ty < M && kid + tx < K) {
      Ashmem[ty][tx] = A[(by * 16 + ty) * K + kid + tx];
    } else {
      Ashmem[ty][tx] = 0.0f;
    }

    // Load B tile
    if (kid + ty < K && bx * 16 + tx < N) {
      Bshmem[ty][tx] = B[(kid + ty) * N + bx * 16 + tx];
    } else {
      Bshmem[ty][tx] = 0.0f;
    }

    __syncthreads();

    // Compute
    for (int i = 0; i < 16; ++i) {
      sum += Ashmem[ty][i] * Bshmem[i][tx];
    }

    __syncthreads();
  }

  // Write output
  if (by * 16 + ty < M && bx * 16 + tx < N) {
    C[(by * 16 + ty) * N + bx * 16 + tx] = sum;
  }
}

torch::Tensor gemm_fp32_hip(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
  int m = a.size(0);
  int n = b.size(1);
  int k = a.size(1);
  dim3 threads(16, 16);
  dim3 blocks((n + 15) / 16, (m + 15) / 16);
  gemm_tiled_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), m, n, k);
  return out;
}
"""

gemm_module = load_inline(
    name="gemm",
    cpp_sources=hip_source,
    functions=["gemm_fp32_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, M: int, K: int, N: int, use_e4m3: bool = True):
        super().__init__()
        self.M = M
        self.K = K
        self.N = N
        self.use_e4m3 = use_e4m3
        self.weight = nn.Parameter(torch.randn(K, N) * 0.02)
        self.gemm = gemm_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x_2d = x.view(-1, self.K).to(torch.float32).contiguous()
        w = self.weight.to(torch.float32).contiguous()
        out_fp32 = torch.empty((x_2d.shape[0], self.N), dtype=torch.float32, device=x.device).contiguous()
        self.gemm.gemm_fp32_hip(x_2d, w, out_fp32)
        return out_fp32.to(input_dtype).view(batch_size, seq_len, self.N)
