
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

rmsnorm_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__global__ void rmsnorm_kernel(const float* __restrict__ x, float* __restrict__ out, int N, int C, int rest, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rest = N * rest;
    if (idx < total_rest) {
        int n = idx / rest;
        int r = idx % rest;
        int base_idx = n * C * rest + r;

        float sum_sq = 0.0f;
        #pragma unroll 4
        for (int c = 0; c < C; ++c) {
            float val = x[base_idx + c * rest];
            sum_sq += val * val;
        }

        float inv_rms = 1.0f / sqrtf(sum_sq / (float)C + eps);

        #pragma unroll 4
        for (int c = 0; c < C; ++c) {
            int x_idx = base_idx + c * rest;
            out[x_idx] = x[x_idx] * inv_rms;
        }
    }
}

torch::Tensor rmsnorm_hip(torch::Tensor x, float eps) {
    auto N = x.size(0);
    auto C = x.size(1);
    auto rest = x.numel() / (N * C);
    auto out = torch::empty_like(x);

    int total_threads = N * rest;
    const int block_size = 256;
    const int num_blocks = (total_threads + block_size - 1) / block_size;

    rmsnorm_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), N, C, rest, eps);

    return out;
}
"""

rmsnorm_lib = load_inline(
    name="rmsnorm_lib",
    cpp_sources=rmsnorm_cpp_source,
    functions=["rmsnorm_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.rmsnorm_lib = rmsnorm_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm_lib.rmsnorm_hip(x, self.eps)
