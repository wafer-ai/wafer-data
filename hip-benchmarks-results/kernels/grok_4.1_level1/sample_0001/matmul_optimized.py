import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

matmul_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define TILE_SZ 64

__global__ void tiled_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SZ][TILE_SZ];
    __shared__ float Bs[TILE_SZ][TILE_SZ];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SZ + ty;
    int col = bx * TILE_SZ + tx;

    float acc = 0.0f;

    int num_tiles = (K + TILE_SZ - 1) / TILE_SZ;
    for (int t = 0; t < num_tiles; ++t) {
        if (row < M && t * TILE_SZ + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SZ + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (col < N && t * TILE_SZ + ty < K) {
            Bs[ty][tx] = B[(t * TILE_SZ + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SZ; ++i) {
            acc += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0);
    auto Kdim = A.size(1);
    auto N = B.size(1);
    int K = Kdim;
    auto options = A.options();
    auto C = torch::zeros({M, N}, options);

    const int TS = TILE_SZ;
    dim3 block(TS, TS);
    dim3 grid((N + TS - 1) / TS, (M + TS - 1) / TS);
    size_t shmem = TS * TS * sizeof(float) * 2;

    tiled_matmul<<<grid, block, shmem>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), (int)M, (int)N, (int)K);

    return C;
}
"""

matmul_module = load_inline(
    name="matmul",
    cpp_sources=matmul_cpp_source,
    functions=["matmul_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_module

    def forward(self, A, B):
        return self.matmul.matmul_hip(A, B)

M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]

def get_init_inputs():
    return []
