import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

gemv_cpp_source = """
#include <hip/hip_runtime.h>

static __inline__ __device__ float warpReduceSum(float val) {
  uint64_t mask = 0xffffffffffffffffULL;
#pragma unroll
  for (int i = 32; i > 0; i >>= 1) {
    float temp = __shfl_xor_sync(mask, val, i);
    val += temp;
  }
  return val;
}

__global__ void gemv_kernel(const float* A, int lda, const float* B, float* C, int M, int K) {
  constexpr int warp_size = 64;
  constexpr int warps_per_row = 16;
  constexpr int threads_per_block = warps_per_row * warp_size;  // 1024
  constexpr int vec_w = 4;
  constexpr int vec_stride = threads_per_block * vec_w;  // 4096

  int row = blockIdx.x;
  if (row >= M) return;

  int tid = threadIdx.x;
  int wid = tid / warp_size;
  int lane = tid % warp_size;

  float sum = 0.0f;
  for (int it = 0; ; ++it) {
    int col = it * vec_stride + tid * vec_w;
    if (col >= K) break;
    int rem = K - col;
    if (rem >= vec_w) {
      const float4* a_ptr = reinterpret_cast<const float4*>(A + row * lda + col);
      const float4* b_ptr = reinterpret_cast<const float4*>(B + col);
      float4 a4 = a_ptr[0];
      float4 b4 = b_ptr[0];
      sum += a4.x * b4.x + a4.y * b4.y + a4.z * b4.z + a4.w * b4.w;
    } else {
      for (int v = 0; v < rem; ++v) {
        sum += A[row * lda + col + v] * B[col + v];
      }
    }
  }

  sum = warpReduceSum(sum);

  __shared__ float warp_partials[16];
  if (lane == 0) {
    warp_partials[wid] = sum;
  }
  __syncthreads();

  if (wid == 0) {
    float total = warp_partials[0];
#pragma unroll
    for (int w = 1; w < warps_per_row; ++w) {
      total += warp_partials[w];
    }
    if (lane == 0) {
      C[row] = total;
    }
  }
}

torch::Tensor gemv_hip(torch::Tensor A, torch::Tensor B) {
  auto M_ = A.size(0);
  auto K_ = A.size(1);
  int M = (int)M_;
  int K = (int)K_;
  int lda = (int)A.stride(0);

  auto options = A.options();
  torch::Tensor C = torch::zeros({M_, 1}, options);

  auto A_ptr = A.data_ptr<float>();
  auto B_ptr = B.data_ptr<float>();
  auto C_ptr = C.data_ptr<float>();

  constexpr int threads_per_block = 1024;
  dim3 threads(threads_per_block);
  dim3 blocks(M);

  gemv_kernel<<<blocks, threads>>>(A_ptr, lda, B_ptr, C_ptr, M, K);
  return C;
}
"""

gemv = load_inline(
    name="gemv",
    cpp_sources=gemv_cpp_source,
    functions=["gemv_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gemv = gemv

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.gemv.gemv_hip(A, B)
