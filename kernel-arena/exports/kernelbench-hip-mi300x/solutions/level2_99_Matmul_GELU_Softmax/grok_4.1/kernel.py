import os
os.environ["CXX"] = "hipcc"
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline
from torch.nn.init import _calculate_fan_in_and_fan_out

batch_size = 1024
in_features = 8192
out_features = 8192

fused_linear_gelu_cpp_source = r'''
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <limits>
#include <cmath>
#include <cfloat>

__device__ float gelu(float x) {
  return 0.5f * x * (1.0f + erf(0.7071067811865475f * x));
}

__global__ void fused_linear_gelu_kernel(const float *input, const float *weight, const float *bias, float *output, int M, int N, int K) {
  constexpr int TILE_M = 64;
  constexpr int TILE_N = 32;
  constexpr int TILE_K = 32;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = by * TILE_M + ty;
  int col = bx * TILE_N + tx;
  if (row >= M || col >= N) return;
  float acc = bias[col];
  __shared__ float As[TILE_M][TILE_K];
  __shared__ float Bs[TILE_K][TILE_N];
  int num_tiles = K / TILE_K;
  for (int t = 0; t < num_tiles; ++t) {
    if (tx < TILE_K) {
      int kidx = t * TILE_K + tx;
      As[ty][tx] = (row < M && kidx < K) ? input[row * K + kidx] : 0.0f;
    }
    if (ty < TILE_K) {
      int kidx = t * TILE_K + ty;
      Bs[ty][tx] = (col < N && kidx < K) ? weight[col * K + kidx] : 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < TILE_K; ++i) {
      acc += As[ty][i] * Bs[i][tx];
    }
    __syncthreads();
  }
  output[row * N + col] = gelu(acc);
}

torch::Tensor fused_linear_gelu_hip(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
  torch::Tensor input_ = input.contiguous();
  torch::Tensor weight_ = weight.contiguous();
  torch::Tensor bias_ = bias.contiguous();
  auto sizes_a = input_.sizes();
  TORCH_CHECK(sizes_a.size() == 2, "Input must be 2D");
  int64_t M = sizes_a[0];
  int64_t K = sizes_a[1];
  int64_t N = weight_.size(0);
  TORCH_CHECK(weight_.size(1) == K, "Weight dim mismatch");
  TORCH_CHECK(bias_.size(0) == N, "Bias dim mismatch");
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input_.device());
  auto out = torch::empty({M, N}, options);
  constexpr int TILE_M = 64;
  constexpr int TILE_N = 32;
  dim3 block(TILE_N, TILE_M);
  dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
  hipLaunchKernelGGL(fused_linear_gelu_kernel, grid, block, 0, 0, input_.data_ptr<float>(), weight_.data_ptr<float>(), bias_.data_ptr<float>(), out.data_ptr<float>(), (int)M, (int)N, (int)K);
  return out;
}
'''

fused_linear_gelu = load_inline(
    name="fused_linear_gelu",
    cpp_sources=fused_linear_gelu_cpp_source,
    functions=["fused_linear_gelu_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty((out_features,), dtype=torch.float32))
        self.reset_parameters()
        self.fused = fused_linear_gelu

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = self.fused.fused_linear_gelu_hip(x, self.weight, self.bias)
        x = F.softmax(x, dim=1)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]
