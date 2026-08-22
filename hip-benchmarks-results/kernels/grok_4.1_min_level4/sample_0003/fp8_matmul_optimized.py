import os
os.environ["CXX"] = "hipcc"

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fp32_gemm_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/torch.h>

__global__ void gemm_kernel(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c, int M, int N, int K) {
  constexpr int TILE_SIZE = 16;
  __shared__ float shared_a[TILE_SIZE][TILE_SIZE];
  __shared__ float shared_b[TILE_SIZE][TILE_SIZE];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * TILE_SIZE + ty;
  int col = bx * TILE_SIZE + tx;

  float sum = 0.0f;
  for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
    if (row < M && (t * TILE_SIZE + tx) < K) {
      shared_a[ty][tx] = a[row * K + t * TILE_SIZE + tx];
    } else {
      shared_a[ty][tx] = 0.0f;
    }
    if (col < N && (t * TILE_SIZE + ty) < K) {
      shared_b[ty][tx] = b[(t * TILE_SIZE + ty) * N + col];
    } else {
      shared_b[ty][tx] = 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
      sum += shared_a[ty][k] * shared_b[k][tx];
    }
    __syncthreads();
  }
  if (row < M && col < N) {
    c[row * N + col] = sum;
  }
}

torch::Tensor fp32_gemm_hip(torch::Tensor a, torch::Tensor b) {
  int M = a.size(0);
  int K = a.size(1);
  int N = b.size(1);
  auto out = torch::empty({M, N}, a.options());
  dim3 block(16, 16);
  dim3 grid((N + 15) / 16, (M + 15) / 16);
  size_t shmem_bytes = 2 * 16 * 16 * sizeof(float);
  hipLaunchKernelGGL(gemm_kernel, grid, block, shmem_bytes, 0,
                     a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), M, N, K);
  return out;
}
"""

fp32_gemm = load_inline(
    name="fp32_gemm",
    cpp_sources=fp32_gemm_cpp_source,
    functions=["fp32_gemm_hip"],
    verbose=True,
)

batch_size = 8
seq_len = 2048
M = batch_size * seq_len
K = 4096
N = 4096
use_e4m3 = True

class ModelNew(nn.Module):
    def __init__(self, M: int, K: int, N: int, use_e4m3: bool = True):
        super().__init__()
        self.M = M
        self.K = K
        self.N = N
        self.use_e4m3 = use_e4m3

        if use_e4m3:
            self.fp8_dtype = torch.float8_e4m3fn
            self.fp8_max = 448.0
        else:
            self.fp8_dtype = torch.float8_e5m2
            self.fp8_max = 57344.0

        self.weight = nn.Parameter(torch.randn(K, N) * 0.02)

        w_scale = self.compute_scale(self.weight)
        self.register_buffer("w_scale", w_scale)

        w_t = self.weight.t().contiguous()
        w_fp8 = self.quantize_to_fp8(w_t, w_scale)
        self.register_buffer("w_fp8", w_fp8)

        w_fp8_float = w_fp8.to(torch.float32)
        w_dequant_t = w_fp8_float * (1.0 / w_scale)
        self.register_buffer("w_dequant", w_dequant_t.t().contiguous())

        self.fp32_gemm_module = fp32_gemm

    def compute_scale(self, x: torch.Tensor) -> torch.Tensor:
        amax = x.abs().max()
        scale = self.fp8_max / amax.clamp(min=1e-12)
        return scale

    def quantize_to_fp8(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        x_scaled = x * scale
        x_clamped = x_scaled.clamp(-self.fp8_max, self.fp8_max)
        return x_clamped.to(self.fp8_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        x_2d = x.view(-1, self.K)

        x_scale = self.compute_scale(x_2d)
        x_fp8 = self.quantize_to_fp8(x_2d, x_scale)
        x_scale_inv = 1.0 / x_scale.to(torch.float32)

        x_dequant = x_fp8.to(torch.float32) * x_scale_inv.contiguous()

        out_fp32 = self.fp32_gemm_module.fp32_gemm_hip(x_dequant.contiguous(), self.w_dequant)

        out = out_fp32.to(input_dtype)
        return out.view(batch_size, seq_len, self.N)

def get_inputs():
    return [torch.randn(batch_size, seq_len, K, dtype=torch.float16).cuda()]

def get_init_inputs():
    return [M, K, N, use_e4m3]
