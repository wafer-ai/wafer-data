import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

hip_source = """
#include <hip/hip_runtime.h>

__global__ void fused_linear_gelu_kernel(const float* __restrict__ x, const float* __restrict__ weight, const float* __restrict__ bias, float* __restrict__ out, int B, int K, int N) {
    constexpr int TILE_M = 32;
    constexpr int TILE_N = 32;
    constexpr int TILE_K = 32;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int m = bx * TILE_M + ty;
    const int n = by * TILE_N + tx;
    if (m >= B || n >= N) {
        return;
    }
    float acc = bias[n];
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];
    for (int kk = 0; kk < K; kk += TILE_K) {
        // Load A tile
        if (tx < TILE_K) {
            int global_k = kk + tx;
            As[ty][tx] = (global_k < K) ? x[m * K + global_k] : 0.0f;
        }
        // Load B tile
        if (tx < TILE_K) {
            int global_k = kk + tx;
            int global_row = by * TILE_N + ty;
            Bs[tx][ty] = (global_k < K) ? weight[global_row * K + global_k] : 0.0f;
        }
        __syncthreads();
        // Compute
        for (int k = 0; k < TILE_K; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    // Gelu
    const float k_b = 0.044715f;
    const float k_a = 0.7978845608f;
    float y = acc + k_b * acc * acc * acc;
    y = k_a * y;
    y = tanhf(y);
    out[m * N + n] = 0.5f * acc * (1.0f + y);
}

torch::Tensor fused_linear_gelu_hip(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    int B = x.size(0);
    int K = x.size(1);
    int N = weight.size(0);
    auto out = torch::empty({B, N}, x.options());
    constexpr int TILE_M = 32;
    constexpr int TILE_N = 32;
    dim3 block(TILE_N, TILE_M);
    dim3 grid((B + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);
    fused_linear_gelu_kernel<<<grid, block>>>(x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), B, K, N);
    return out;
}
"""

fused_linear_gelu = load_inline(
    name="fused_linear_gelu",
    cpp_sources=hip_source,
    functions=["fused_linear_gelu_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias, -bound, bound)
        self.fused_linear_gelu = fused_linear_gelu

    def forward(self, x):
        gelu_out = self.fused_linear_gelu.fused_linear_gelu_hip(x, self.weight, self.bias)
        return F.softmax(gelu_out, dim=1)


batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]
