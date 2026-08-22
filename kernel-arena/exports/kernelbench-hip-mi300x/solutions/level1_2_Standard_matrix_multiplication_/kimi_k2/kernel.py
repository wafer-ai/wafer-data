import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

matmul_cpp_source = """
#include <hip/hip_runtime.h>

#define TILE_M 128
#define TILE_N 128
#define TILE_K 32
#define THREAD_M 8
#define THREAD_N 8

__global__ void matmul_optimized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    // Shared memory for tiles - use 2D layout to reduce bank conflicts
    __shared__ float tile_A[TILE_M][TILE_K];
    __shared__ float tile_B[TILE_K][TILE_N];
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int thread_id = ty * blockDim.x + tx;
    
    // Block indices
    int block_row = blockIdx.y * TILE_M;
    int block_col = blockIdx.x * TILE_N;
    
    // Local register file for accumulation - 8x8 per thread
    float acc[THREAD_M][THREAD_N] = {{0.0f}};
    
    // Loop over tiles of K dimension
    for (int tile_k = 0; tile_k < K; tile_k += TILE_K) {
        // Load tile of A into shared memory - coalesced access
        int num_threads = blockDim.x * blockDim.y;
        for (int i = thread_id; i < TILE_M * TILE_K; i += num_threads) {
            int shared_row = i / TILE_K;
            int shared_col = i % TILE_K;
            int global_row = block_row + shared_row;
            int global_col = tile_k + shared_col;
            
            if (global_row < M && global_col < K) {
                tile_A[shared_row][shared_col] = A[global_row * K + global_col];
            } else {
                tile_A[shared_row][shared_col] = 0.0f;
            }
        }
        
        // Load tile of B into shared memory - coalesced access
        for (int i = thread_id; i < TILE_K * TILE_N; i += num_threads) {
            int shared_row = i / TILE_N;
            int shared_col = i % TILE_N;
            int global_row = tile_k + shared_row;
            int global_col = block_col + shared_col;
            
            if (global_row < K && global_col < N) {
                tile_B[shared_row][shared_col] = B[global_row * N + global_col];
            } else {
                tile_B[shared_row][shared_col] = 0.0f;
            }
        }
        
        // Synchronize to ensure tiles are loaded
        __syncthreads();
        
        // Each thread computes THREAD_M x THREAD_N block using the loaded tiles
        int thread_row_base = (thread_id / (TILE_N / THREAD_N)) * THREAD_M;
        int thread_col_base = (thread_id % (TILE_N / THREAD_N)) * THREAD_N;
        
        // Compute partial product - manual loop unrolling for better performance
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            #pragma unroll
            for (int i = 0; i < THREAD_M; ++i) {
                float a_val = tile_A[thread_row_base + i][k];
                #pragma unroll
                for (int j = 0; j < THREAD_N; ++j) {
                    acc[i][j] += a_val * tile_B[k][thread_col_base + j];
                }
            }
        }
        
        // Synchronize before loading next tiles
        __syncthreads();
    }
    
    // Write results to C
    int thread_row_base = (thread_id / (TILE_N / THREAD_N)) * THREAD_M;
    int thread_col_base = (thread_id % (TILE_N / THREAD_N)) * THREAD_N;
    
    for (int i = 0; i < THREAD_M; ++i) {
        int global_row = block_row + thread_row_base + i;
        if (global_row >= M) continue;
        
        for (int j = 0; j < THREAD_N; ++j) {
            int global_col = block_col + thread_col_base + j;
            if (global_col >= N) continue;
            
            C[global_row * N + global_col] = acc[i][j];
        }
    }
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda(), "A must be on CUDA device");
    TORCH_CHECK(B.device().is_cuda(), "B must be on CUDA device");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be float32");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    TORCH_CHECK(B.size(0) == K, "Incompatible dimensions");
    
    auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(A.device()));
    
    // Launch configuration: 256 threads per block
    int threads = 256;
    dim3 block(threads / 4, 4);  // 64x4 gives better SM utilization
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    
    matmul_optimized_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );
    
    return C;
}
"""

matmul = load_inline(
    name="matmul",
    cpp_sources=matmul_cpp_source,
    functions=["matmul_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = matmul
    
    def forward(self, A, B):
        return self.matmul.matmul_hip(A, B)


M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []
