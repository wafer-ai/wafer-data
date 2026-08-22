import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__global__ void rmsnorm_kernel(const float* __restrict__ x, float* __restrict__ out, float eps, int64_t N, int64_t F_, int64_t H, int64_t W, int64_t stride_f) {
    uint32_t tid = threadIdx.x;
    uint32_t n_loc = blockIdx.x;
    uint32_t h_loc = blockIdx.y;
    uint32_t w_loc = blockIdx.z;
    int64_t f = static_cast<int64_t>(tid);
    if (f >= F_) return;
    int64_t n = static_cast<int64_t>(n_loc);
    int64_t h = static_cast<int64_t>(h_loc);
    int64_t w = static_cast<int64_t>(w_loc);
    if (n >= N || h >= H || w >= W) return;
    int64_t idx = n * (F_ * stride_f) + f * stride_f + h * W + w;
    uint64_t mask = 0xffffffffffffffffULL;
    float sum_sq = x[idx] * x[idx];
    sum_sq += __shfl_xor_sync(mask, sum_sq, 32);
    sum_sq += __shfl_xor_sync(mask, sum_sq, 16);
    sum_sq += __shfl_xor_sync(mask, sum_sq, 8);
    sum_sq += __shfl_xor_sync(mask, sum_sq, 4);
    sum_sq += __shfl_xor_sync(mask, sum_sq, 2);
    sum_sq += __shfl_xor_sync(mask, sum_sq, 1);
    float rms = sqrtf(sum_sq / static_cast<float>(F_) + eps);
    float val = x[idx];
    out[idx] = val / rms;
}

torch::Tensor rmsnorm_hip(torch::Tensor x, float eps) {
    auto N_ = x.size(0);
    auto F_ = x.size(1);
    auto H_ = x.size(2);
    auto W_ = x.size(3);
    auto out = torch::empty_like(x);
    int64_t stride_f = H_ * W_;
    dim3 block(static_cast<unsigned int>(F_));
    dim3 grid(static_cast<unsigned int>(N_), static_cast<unsigned int>(H_), static_cast<unsigned int>(W_));
    hipLaunchKernelGGL(rmsnorm_kernel, grid, block, 0, 0,
                       x.data_ptr<float>(), out.data_ptr<float>(), eps,
                       N_, F_, H_, W_, stride_f);
    return out;
}
"""

rmsnorm = load_inline(
    name="rmsnorm",
    cpp_sources=hip_source,
    functions=["rmsnorm_hip"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm.rmsnorm_hip(x, self.eps)
