import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

matmul_cpp = """
#include <hip/hip_runtime.h>

__global__ void matmul_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    const int TS = 32;
    __shared__ float As[TS][TS];
    __shared__ float Bs[TS][TS];
    int row = by * TS + ty;
    int col = bx * TS + tx;
    float acc = 0.0f;
    const int NUM_TILES = (K + TS - 1) / TS;
    for (int ph = 0; ph < NUM_TILES; ++ph) {
        As[ty][tx] = (row < M && ph * TS + tx < K) ? A[row * K + ph * TS + tx] : 0.0f;
        Bs[ty][tx] = (col < N && ph * TS + ty < K) ? B[(ph * TS + ty) * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TS; ++i) {
            acc += As[ty][i] * Bs[i][tx];
        }
        // Removed syncthreads here
    }
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

torch::Tensor matmul_hip(torch::Tensor a, torch::Tensor b) {
    int64_t M = a.sizes()[0];
    int64_t K = a.sizes()[1];
    int64_t N = b.sizes()[1];
    torch::Tensor c = torch::zeros({M, N}, a.options());
    const int TS = 32;
    dim3 block(TS, TS);
    dim3 grid((N + TS - 1) / TS, (M + TS - 1) / TS);
    matmul_kernel<<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), (int)M, (int)N, (int)K);
    return c;
}
"""

matmul_ext = load_inline(
    name="matmul",
    cpp_sources=matmul_cpp,
    functions=["matmul_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_hip = matmul_ext

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_hip.matmul_hip(A, B)

N = 2048 * 2

def get_inputs():
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    return [A, B]

def get_init_inputs():
    return []
