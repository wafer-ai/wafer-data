import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Matrix Multiplication Kernel for MI300X
# Optimized tiled implementation with shared memory
matmul_hip_source = """
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define BLOCK_SIZE_M 128
#define BLOCK_SIZE_N 128
#define BLOCK_SIZE_K 16
#define THREAD_SIZE_M 8
#define THREAD_SIZE_N 8

__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N) {
    
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Thread index within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Calculate thread ID for 1D indexing
    int tid = tx + ty * blockDim.x;
    
    // Shared memory for tiles
    __shared__ float As[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
    
    // Calculate number of tiles in K dimension
    int num_tiles = (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K;
    
    // Thread-local accumulator registers
    float c_reg[THREAD_SIZE_M][THREAD_SIZE_N];
    #pragma unroll
    for (int i = 0; i < THREAD_SIZE_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_SIZE_N; j++) {
            c_reg[i][j] = 0.0f;
        }
    }
    
    // Calculate starting positions for this block
    int row_start_a = by * BLOCK_SIZE_M;
    int col_start_b = bx * BLOCK_SIZE_N;
    
    // Loop over tiles in K dimension
    for (int t = 0; t < num_tiles; t++) {
        // Load tile from A into shared memory
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE_M; i += blockDim.y) {
            int global_row = row_start_a + i + ty;
            int global_col = t * BLOCK_SIZE_K + tx;
            if (global_row < M && global_col < K) {
                As[i + ty][tx] = A[global_row * K + global_col];
            } else {
                As[i + ty][tx] = 0.0f;
            }
        }
        
        // Load tile from B into shared memory
        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE_N; j += blockDim.x) {
            int global_row = t * BLOCK_SIZE_K + ty;
            int global_col = col_start_b + j + tx;
            if (global_row < K && global_col < N) {
                Bs[ty][j + tx] = B[global_row * N + global_col];
            } else {
                Bs[ty][j + tx] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute using shared memory tiles
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; k++) {
            // Each thread loads one row from As and one column from Bs
            // and performs THREAD_SIZE_M * THREAD_SIZE_N multiply-add operations
            float a_vals[THREAD_SIZE_M];
            float b_vals[THREAD_SIZE_N];
            
            #pragma unroll
            for (int i = 0; i < THREAD_SIZE_M; i++) {
                int row_a = ty * THREAD_SIZE_M + i;
                a_vals[i] = As[row_a][k];
            }
            
            #pragma unroll
            for (int j = 0; j < THREAD_SIZE_N; j++) {
                int col_b = tx * THREAD_SIZE_N + j;
                b_vals[j] = Bs[k][col_b];
            }
            
            #pragma unroll
            for (int i = 0; i < THREAD_SIZE_M; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_SIZE_N; j++) {
                    c_reg[i][j] += a_vals[i] * b_vals[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results to global memory
    #pragma unroll
    for (int i = 0; i < THREAD_SIZE_M; i++) {
        int global_row = row_start_a + ty * THREAD_SIZE_M + i;
        if (global_row < M) {
            #pragma unroll
            for (int j = 0; j < THREAD_SIZE_N; j++) {
                int global_col = col_start_b + tx * THREAD_SIZE_N + j;
                if (global_col < N) {
                    C[global_row * N + global_col] = c_reg[i][j];
                }
            }
        }
    }
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    // Get dimensions
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    // Create output tensor
    auto C = torch::zeros({M, N}, A.options());
    
    // Calculate grid and block dimensions
    dim3 block_dim(16, 16);  // 256 threads per block
    dim3 grid_dim((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, 
                  (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
    
    // Launch kernel
    hipLaunchKernelGGL(matmul_kernel, grid_dim, block_dim, 0, 0,
                       A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
                       M, K, N);
    
    return C;
}
"""

matmul_hip = load_inline(
    name="matmul_hip",
    cpp_sources=matmul_hip_source,
    functions=["matmul_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return matmul_hip.matmul_hip(A.cuda(), B.cuda())

M = 8205
K = 2949
N = 5921

def get_inputs():
    A = torch.rand(M, K, device='cuda')
    B = torch.rand(K, N, device='cuda')
    return [A, B]

def get_init_inputs():
    return []