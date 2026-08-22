import os
os.environ['CXX'] = 'hipcc'
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_cpp = '''
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void sgemm_tiled(const float *__restrict__ A_global, const float *__restrict__ B_global, float *__restrict__ C_global, const int M, const int N, const int K) {
    const int TILE_M = 32;
    const int TILE_N = 32;
    const int TILE_K = 64;
    extern __shared__ float shmem[];
    float *As = shmem;
    float *Bs = As + TILE_M * TILE_K;
    int Row = blockIdx.y * TILE_M + threadIdx.y;
    int Col = blockIdx.x * TILE_N + threadIdx.x;
    float Pvalue = 0.0f;
    const int num_tiles = (K + TILE_K - 1) / TILE_K;
    for (int m = 0; m < num_tiles; ++m) {
        // Load A tile
        for (int j = 0; j < TILE_K / blockDim.x; ++j) {
            int k_load = j * blockDim.x + threadIdx.x;
            if (Row < M && m * TILE_K + k_load < K) {
                As[threadIdx.y * TILE_K + k_load] = A_global[Row * K + m * TILE_K + k_load];
            } else {
                As[threadIdx.y * TILE_K + k_load] = 0.0f;
            }
        }
        // Load B tile
        for (int j = 0; j < TILE_K / blockDim.y; ++j) {
            int k_load = j * blockDim.y + threadIdx.y;
            if (Col < N && m * TILE_K + k_load < K) {
                Bs[k_load * TILE_N + threadIdx.x] = B_global[(m * TILE_K + k_load) * N + Col];
            } else {
                Bs[k_load * TILE_N + threadIdx.x] = 0.0f;
            }
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            Pvalue += As[threadIdx.y * TILE_K + k] * Bs[k * TILE_N + threadIdx.x];
        }
    }
    if (Row < M && Col < N) {
        C_global[Row * N + Col] = Pvalue;
    }
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    int64_t M_ = A.size(0);
    int64_t K_ = A.size(1);
    int64_t N_ = B.size(1);
    int M = static_cast<int>(M_);
    int K = static_cast<int>(K_);
    int N = static_cast<int>(N_);
    auto C = torch::zeros({M_, N_}, A.options());
    const int TILE_M = 32;
    const int TILE_N = 32;
    const int TILE_K = 64;
    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    size_t shmem_bytes = (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(float);
    hipLaunchKernelGGL(sgemm_tiled, grid, block, shmem_bytes, 0, A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}
'''

matmul_ext = load_inline(
    name='matmul',
    cpp_sources=matmul_cpp,
    functions=['matmul_hip'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matmul = matmul_ext

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.matmul_hip(A, B)
