import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// For tall-skinny matmul where K is small (32)
// A is M x K, B is K x M, C is M x M
// Strategy: tile in M dimension, process many output rows sharing the same B data

#define TILE_M 16
#define TILE_N 64
#define K_SIZE 32

__global__ void tall_skinny_matmul_kernel_v5(
    const float* __restrict__ A,  // M x K
    const float* __restrict__ B,  // K x M
    float* __restrict__ C,        // M x M
    int M, int K
) {
    // Shared memory for B tile
    __shared__ float Bs[K_SIZE][TILE_N + 1];  // +1 to avoid bank conflicts
    __shared__ float As[TILE_M][K_SIZE + 1];
    
    int block_row = blockIdx.y * TILE_M;
    int block_col = blockIdx.x * TILE_N;
    
    // Thread indexing: 16x16 = 256 threads
    int tx = threadIdx.x;  // 0-15
    int ty = threadIdx.y;  // 0-15
    int tid = ty * blockDim.x + tx;
    
    // Load B tile into shared memory (K x TILE_N)
    // K = 32, TILE_N = 64, so 2048 elements, 256 threads -> 8 elements per thread
    for (int i = tid; i < K * TILE_N; i += 256) {
        int k = i / TILE_N;
        int n = i % TILE_N;
        int global_col = block_col + n;
        if (k < K && global_col < M) {
            Bs[k][n] = B[k * M + global_col];
        } else {
            Bs[k][n] = 0.0f;
        }
    }
    
    // Load A tile into shared memory (TILE_M x K)
    // TILE_M = 16, K = 32, so 512 elements, 256 threads -> 2 elements per thread
    for (int i = tid; i < TILE_M * K; i += 256) {
        int m = i / K;
        int k = i % K;
        int global_row = block_row + m;
        if (global_row < M && k < K) {
            As[m][k] = A[global_row * K + k];
        } else {
            As[m][k] = 0.0f;
        }
    }
    
    __syncthreads();
    
    // Each thread computes one element of output
    // We have 16x16 = 256 threads, need to compute TILE_M x TILE_N = 16 x 64 = 1024 elements
    // So each thread computes 4 elements (1 row, 4 columns)
    for (int out_idx = tid; out_idx < TILE_M * TILE_N; out_idx += 256) {
        int local_row = out_idx / TILE_N;
        int local_col = out_idx % TILE_N;
        int global_row = block_row + local_row;
        int global_col = block_col + local_col;
        
        if (global_row < M && global_col < M) {
            float sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < K_SIZE; ++k) {
                if (k < K) {
                    sum += As[local_row][k] * Bs[k][local_col];
                }
            }
            C[global_row * M + global_col] = sum;
        }
    }
}

torch::Tensor tall_skinny_matmul_hip(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int M2 = B.size(1);
    
    auto C = torch::empty({M, M2}, A.options());
    
    dim3 block(16, 16);
    dim3 grid((M2 + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    
    tall_skinny_matmul_kernel_v5<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K
    );
    
    return C;
}
"""

cpp_source = """
torch::Tensor tall_skinny_matmul_hip(torch::Tensor A, torch::Tensor B);
"""

tall_skinny_matmul = load_inline(
    name="tall_skinny_matmul",
    cpp_sources=cpp_source,
    cuda_sources=hip_source,
    functions=["tall_skinny_matmul_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tall_skinny_matmul = tall_skinny_matmul
    
    def forward(self, A, B):
        return self.tall_skinny_matmul.tall_skinny_matmul_hip(A, B)


def get_inputs():
    M = 16384 * 2
    N = 16 * 2
    A = torch.rand(M, N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]


def get_init_inputs():
    return []
