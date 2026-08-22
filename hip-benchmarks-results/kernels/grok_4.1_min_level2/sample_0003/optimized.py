import os
import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

fused_cpp = r"""
#include <hip/hip_runtime.h>

__global__ void fused_tiled_kernel(const float *x, const float *w, const float *b, float *out, float sf, int B, int K, int N) {
    constexpr int TM = 32;
    constexpr int TN = 32;
    constexpr int TK = 128;
    int tm = blockIdx.y;
    int tn = blockIdx.x;
    int tx = threadIdx.x;  // local_n
    int ty = threadIdx.y;  // local_m
    int m = tm * TM + ty;
    int n = tn * TN + tx;
    if (m >= B || n >= N) return;
    float acc = 0.0f;
    __shared__ float Ash[TM][TK];
    __shared__ float Bsh[TK][TN];
    int x_off = m * K;
    for (int tk = 0; tk < K; tk += TK) {
        // Load A tile
        int kbase = tk + tx * 4;
        Ash[ty][kbase + 0] = x[x_off + kbase + 0];
        Ash[ty][kbase + 1] = x[x_off + kbase + 1];
        Ash[ty][kbase + 2] = x[x_off + kbase + 2];
        Ash[ty][kbase + 3] = x[x_off + kbase + 3];
        // Load B tile
        int wrow = (tn * TN + ty);
        int wkbase = tk + tx * 4;
        Bsh[kbase + 0][ty] = w[wrow * K + wkbase + 0];
        Bsh[kbase + 1][ty] = w[wrow * K + wkbase + 1];
        Bsh[kbase + 2][ty] = w[wrow * K + wkbase + 2];
        Bsh[kbase + 3][ty] = w[wrow * K + wkbase + 3];
        __syncthreads();
        // Compute
        for (int kk = 0; kk < TK; ++kk) {
            acc += Ash[ty][kk] * Bsh[kk][tx];
        }
        __syncthreads();
    }
    acc += b[n];
    acc *= (sf + 1.0f);
    out[m * N + n] = acc;
}

torch::Tensor fused_linear_residual_scale_hip(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float sf) {
    auto B64 = x.size(0);
    auto K64 = x.size(1);
    auto N64 = weight.size(0);
    int B = static_cast<int>(B64);
    int K = static_cast<int>(K64);
    int N = static_cast<int>(N64);
    torch::Tensor outt = torch::empty({B64, N64}, x.options());
    const float *x_ptr = x.data_ptr<float>();
    const float *w_ptr = weight.data_ptr<float>();
    const float *b_ptr = bias.data_ptr<float>();
    float *out_ptr = outt.data_ptr<float>();
    constexpr int TM = 32;
    constexpr int TN = 32;
    int num_tm = (B + TM - 1) / TM;
    int num_tn = (N + TN - 1) / TN;
    dim3 block(TN, TM);
    dim3 grid(num_tn, num_tm);
    fused_tiled_kernel<<<grid, block>>>(x_ptr, w_ptr, b_ptr, out_ptr, sf, B, K, N);
    return outt;
}
"""

fused_linear = load_inline(
    name="fused_linear_residual",
    cpp_sources=fused_cpp,
    functions=["fused_linear_residual_scale_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()
        self.fused = fused_linear

    def reset_parameters(self):
        fan_in = self.in_features
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return self.fused.fused_linear_residual_scale_hip(x, self.weight, self.bias, self.scaling_factor)
