import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

M = 8205
K = 2949
N = 5921

tiled_gemm_cpp = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define TILE_SIZE 32
#define K_PER_THREAD 2
#define K_TILE (TILE_SIZE * K_PER_THREAD)

__shared__ float Ashare[TILE_SIZE][K_TILE];
__shared__ float Bshare[K_TILE][TILE_SIZE];

__global__ void tiled_gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Row = blockIdx.y * TILE_SIZE + ty;
    int Col = blockIdx.x * TILE_SIZE + tx;

    float Pvalue = 0.0f;

    int num_tiles = (K + K_TILE - 1) / K_TILE;
    for (int m = 0; m < num_tiles; ++m) {
        // Load A tile
        for(int j = 0; j < K_PER_THREAD; ++j) {
            int kk = m * K_TILE + tx + j * TILE_SIZE;
            if (Row < M && kk < K) {
                Ashare[ty][tx + j * TILE_SIZE] = A[Row * K + kk];
            } else {
                Ashare[ty][tx + j * TILE_SIZE] = 0.0f;
            }
        }

        // Load B tile
        for(int j = 0; j < K_PER_THREAD; ++j) {
            int kk = m * K_TILE + ty + j * TILE_SIZE;
            if (Col < N && kk < K) {
                Bshare[ty + j * TILE_SIZE][tx] = B[kk * N + Col];
            } else {
                Bshare[ty + j * TILE_SIZE][tx] = 0.0f;
            }
        }

        __syncthreads();

        // Compute
        for(int j = 0; j < K_PER_THREAD; ++j) {
            int kk_start = j * TILE_SIZE;
#pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                Pvalue += Ashare[ty][kk_start + k] * Bshare[kk_start + k][tx];
            }
        }

        __syncthreads();
    }

    if (Row < M && Col < N) {
        C[Row * N + Col] = Pvalue;
    }
}

torch::Tensor tiled_matmul_hip(torch::Tensor A, torch::Tensor B) {
    int64_t MM = A.size(0);
    int64_t KK = A.size(1);
    int64_t NN = B.size(1);
    int M = (int)MM;
    int N = (int)NN;
    int K = (int)KK;
    auto C = torch::zeros({MM, NN}, A.options());
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    tiled_gemm_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}
"""

tiled_matmul = load_inline(
    name="tiled_matmul",
    cpp_sources=tiled_gemm_cpp,
    functions=["tiled_matmul_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tiled_matmul = tiled_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.tiled_matmul.tiled_matmul_hip(A, B)

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]

def get_init_inputs():
    return []
