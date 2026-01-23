
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

# Advanced optimized GEMM using shared memory and register tiling
# Optimized for MI300X with larger tiles and better memory access patterns.
gemm_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define TILE_M 128
#define TILE_N 128
#define TILE_K 16
#define MICRO_M 8
#define MICRO_N 8

__global__ void __launch_bounds__(256) gemm_optimized_v2_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    __shared__ float sA[TILE_M][TILE_K];
    __shared__ float sB[TILE_K][TILE_N];

    float rC[MICRO_M][MICRO_N];
    for(int i=0; i<MICRO_M; ++i) for(int j=0; j<MICRO_N; ++j) rC[i][j] = 0.0f;
    
    float rA[MICRO_M];
    float rB[MICRO_N];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x % (TILE_N / MICRO_N);
    int ty = threadIdx.x / (TILE_N / MICRO_N);

    int thread_row = ty * MICRO_M;
    int thread_col = tx * MICRO_N;

    for (int k_outer = 0; k_outer < N; k_outer += TILE_K) {
        // Load A into shared memory
        for (int i = 0; i < (TILE_M * TILE_K) / 256; ++i) {
            int local_id = i * 256 + threadIdx.x;
            int row = local_id / TILE_K;
            int col = local_id % TILE_K;
            sA[row][col] = A[(by * TILE_M + row) * N + (k_outer + col)];
        }

        // Load B into shared memory
        for (int i = 0; i < (TILE_K * TILE_N) / 256; ++i) {
            int local_id = i * 256 + threadIdx.x;
            int row = local_id / TILE_N;
            int col = local_id % TILE_N;
            sB[row][col] = B[(k_outer + row) * N + (bx * TILE_N + col)];
        }

        __syncthreads();

        #pragma unroll
        for (int k_inner = 0; k_inner < TILE_K; ++k_inner) {
            #pragma unroll
            for (int i = 0; i < MICRO_M; ++i) rA[i] = sA[thread_row + i][k_inner];
            #pragma unroll
            for (int j = 0; j < MICRO_N; ++j) rB[j] = sB[k_inner][thread_col + j];

            #pragma unroll
            for (int i = 0; i < MICRO_M; ++i) {
                #pragma unroll
                for (int j = 0; j < MICRO_N; ++j) {
                    rC[i][j] += rA[i] * rB[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < MICRO_M; ++i) {
        #pragma unroll
        for (int j = 0; j < MICRO_N; ++j) {
            C[(by * TILE_M + thread_row + i) * N + (bx * TILE_N + thread_col + j)] = rC[i][j];
        }
    }
}

torch::Tensor gemm_optimized_v2_hip(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::empty({N, N}, A.options());

    dim3 threads(256);
    dim3 blocks(N / TILE_N, N / TILE_M);

    gemm_optimized_v2_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}
"""

gemm_module = load_inline(
    name="gemm_optimized_v2",
    cpp_sources=gemm_cpp_source,
    functions=["gemm_optimized_v2_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gemm = gemm_module

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.gemm.gemm_optimized_v2_hip(A, B)

def get_inputs():
    N = 4096
    A = torch.rand(N, N).cuda()
    B = torch.rand(N, N).cuda()
    return [A, B]

def get_init_inputs():
    return []
