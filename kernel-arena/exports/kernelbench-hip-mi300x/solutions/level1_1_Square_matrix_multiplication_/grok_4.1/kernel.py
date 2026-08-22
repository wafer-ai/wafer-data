import os
os.environ["CXX"] = "hipcc"

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

gemm_cpp = """
#include <hip/hip_runtime.h>

__global__ void gemm_simple(const float *A, const float *B, float *C, int M, int N, int K) {
    const int TS = 32;
    const int TK = 128;
    __shared__ float Ash[TS][TK];
    __shared__ float Bsh[TK][TS];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;
    int row = by * TS + ty;
    int col = bx * TS + tx;

    const int n_phases = TK / TS;

    for (int kt = 0; kt < K; kt += TK) {
        // Load A tile - multiple phases
#pragma unroll
        for (int phase = 0; phase < n_phases; ++phase) {
            int kidx = phase * TS + tx;
            int gk = kt + kidx;
            if (row < M && gk < K) {
                Ash[ty][kidx] = A[row * K + gk];
            } else {
                Ash[ty][kidx] = 0.0f;
            }
        }

        // Load B tile - multiple phases
#pragma unroll
        for (int phase = 0; phase < n_phases; ++phase) {
            int kidx = phase * TS + ty;
            int gk = kt + kidx;
            if (col < N && gk < K) {
                Bsh[kidx][tx] = B[gk * N + col];
            } else {
                Bsh[kidx][tx] = 0.0f;
            }
        }

        __syncthreads();

        // Compute
        for (int kk = 0; kk < TK; ++kk) {
            sum += Ash[ty][kk] * Bsh[kk][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor gemm_hip(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0);
    auto N = B.size(1);
    auto K = A.size(1);
    auto C = torch::zeros({M, N}, A.options());

    const int TS = 32;
    dim3 threads(TS, TS);
    dim3 blocks((N + TS - 1) / TS, (M + TS - 1) / TS);

    gemm_simple<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}
"""

gemm = load_inline(
    name="gemm",
    cpp_sources=gemm_cpp,
    functions=["gemm_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gemm_hip = gemm.gemm_hip

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.gemm_hip(A, B)

N = 2048 * 2

def get_inputs():
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    return [A, B]

def get_init_inputs():
    return []
