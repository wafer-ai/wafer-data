import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>

__global__ void row_reduce_max(const float* input, float* output, int N) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int row = blockIdx.x;
  float val = -3.402823466e+38F;
  for (int i = tid; i < N; i += blockDim.x) {
    val = fmaxf(val, input[row * N + i]);
  }
  sdata[tid] = val;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    output[row] = sdata[0];
  }
}

__global__ void row_reduce_exp_sum(const float* x, const float* row_max, float* output, int N) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int row = blockIdx.x;
  float m = row_max[row];
  float val = 0.0f;
  for (int i = tid; i < N; i += blockDim.x) {
    val += expf(x[row * N + i] - m);
  }
  sdata[tid] = val;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    output[row] = sdata[0];
  }
}

__global__ void normalize_kernel(const float* x, const float* row_max, const float* row_sum, float* output, int B, int N) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)B * (size_t)N) return;
  int row = idx / N;
  float m = row_max[row];
  float s = row_sum[row];
  output[idx] = expf(x[idx] - m) / s;
}

torch::Tensor custom_softmax_hip(torch::Tensor x) {
  int64_t B64 = x.size(0);
  int64_t N64 = x.size(1);
  int B = static_cast<int>(B64);
  int N = static_cast<int>(N64);
  int64_t total = x.numel();
  auto options = x.options();
  auto row_max = torch::empty(torch::IntArrayRef({B64}), options);
  auto row_sum = torch::empty(torch::IntArrayRef({B64}), options);
  auto out = torch::empty_like(x);
  const int reduce_bs = 512;
  const int pw_bs = 1024;
  int pw_grid = static_cast<int>((total + pw_bs - 1LL) / pw_bs);
  dim3 reduce_grid(B);
  dim3 pw_grid_d(pw_grid);
  dim3 reduce_threads(reduce_bs);
  dim3 pw_threads(pw_bs);
  size_t shmem_bytes = reduce_bs * sizeof(float);
  hipLaunchKernelGGL(row_reduce_max, reduce_grid, reduce_threads, shmem_bytes, 0,
                     x.data_ptr<float>(), row_max.data_ptr<float>(), N);
  hipLaunchKernelGGL(row_reduce_exp_sum, reduce_grid, reduce_threads, shmem_bytes, 0,
                     x.data_ptr<float>(), row_max.data_ptr<float>(), row_sum.data_ptr<float>(), N);
  hipLaunchKernelGGL(normalize_kernel, pw_grid_d, pw_threads, 0, 0,
                     x.data_ptr<float>(), row_max.data_ptr<float>(), row_sum.data_ptr<float>(), out.data_ptr<float>(), B, N);
  (void)hipDeviceSynchronize();
  return out;
}
"""

softmax_ext = load_inline(
    name="softmax_hip",
    cpp_sources=cpp_source,
    functions=["custom_softmax_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.custom_softmax = softmax_ext.custom_softmax_hip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_softmax(x)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
