import os
os.environ["CXX"] = "hipcc"

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

cpp_source = """
#include <hip/hip_runtime.h>

__device__ float gelu(float x) {
    constexpr float k1 = 0.7978845608f;
    constexpr float k2 = 0.044715f;
    float x3 = x * x * x;
    float inner = k1 * (x + k2 * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void tiled_fused_linear_div_gelu_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float divisor,
    float* out,
    int B, int N, int K
) {
    constexpr int TM = 16;
    constexpr int TN = 64;
    constexpr int TK = 64;
    constexpr int BS = TM * TN;
    __shared__ float As[TM][TK];
    __shared__ float Bs[TK][TN];

    int b_start = blockIdx.y * TM;
    int n_start = blockIdx.x * TN;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * TN + tx;
    int b = b_start + ty;
    int n = n_start + tx;
    if (b >= B || n >= N) return;

    float accum = bias[n];

    for (int tk = 0; tk < K; tk += TK) {
        // Load A one phase
        int lid = tid;
        if (lid < TM * TK) {
            int row = lid / TK;
            int col = lid % TK;
            As[row][col] = x[(b_start + row) * K + tk + col];
        }
        // Load B unrolled 4 phases
        lid = tid;
        if (lid < TK * TN) {
            int row_b = lid / TN;
            int col_b = lid % TN;
            int g_n = n_start + col_b;
            Bs[row_b][col_b] = weight[g_n * K + tk + row_b];
        }
        lid += BS;
        if (lid < TK * TN) {
            int row_b = lid / TN;
            int col_b = lid % TN;
            int g_n = n_start + col_b;
            Bs[row_b][col_b] = weight[g_n * K + tk + row_b];
        }
        lid += BS;
        if (lid < TK * TN) {
            int row_b = lid / TN;
            int col_b = lid % TN;
            int g_n = n_start + col_b;
            Bs[row_b][col_b] = weight[g_n * K + tk + row_b];
        }
        lid += BS;
        if (lid < TK * TN) {
            int row_b = lid / TN;
            int col_b = lid % TN;
            int g_n = n_start + col_b;
            Bs[row_b][col_b] = weight[g_n * K + tk + row_b];
        }
        __syncthreads();

        int kmax = (TK < K - tk) ? TK : K - tk;
        for (int k = 0; k < kmax; ++k) {
            accum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    out[b * N + n] = gelu(accum / divisor);
}

torch::Tensor fused_linear_div_gelu_hip(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double divisor
) {
    int B = x.size(0);
    int K = x.size(1);
    int N = weight.size(0);
    torch::Tensor out = torch::empty({B, N}, x.options());
    float div = static_cast<float>(divisor);
    constexpr int TM = 16;
    constexpr int TN = 64;
    dim3 block(TN, TM);
    dim3 grid((N + TN - 1) / TN, (B + TM - 1) / TM);
    tiled_fused_linear_div_gelu_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        div,
        out.data_ptr<float>(),
        B, N, K
    );
    return out;
}
"""

custom_fused = load_inline(
    name="fused",
    cpp_sources=cpp_source,
    functions=["fused_linear_div_gelu_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((output_size, input_size)))
        self.bias = nn.Parameter(torch.empty((output_size,)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)
        self.divisor = divisor

    def forward(self, x):
        return custom_fused.fused_linear_div_gelu_hip(x, self.weight, self.bias, self.divisor)
