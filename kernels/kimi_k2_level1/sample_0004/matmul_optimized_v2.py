import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Improved Matrix Multiplication Kernel for MI300X
# Optimized with better memory access patterns and block configuration
matmul_hip_source_v2 = """
#include <hip/hip_runtime.h>

#define BLOCK_SIZE_M 128
#define BLOCK_SIZE_N 128
#define BLOCK_SIZE_K 32

__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N) {
    
    // Cache blocking: each thread computes one 4x4 tile
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int lane_id = tx;
    
    // Global position in output matrix
    int row_start = blockIdx.y * BLOCK_SIZE_M + ty * 4;
    int col_start = blockIdx.x * BLOCK_SIZE_N + lane_id * 4;
    
    // Shared memory for tiles
    __shared__ float As[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
    
    // Thread-local accumulators
    float c_reg[16] = {0.0f};
    
    // Number of tiles in K dimension
    int num_tiles = (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K;
    
    // Loop over tiles in K dimension
    for (int t = 0; t < num_tiles; t++) {
        // Load tile from A into shared memory (coalesced access)
        int a_global_col = t * BLOCK_SIZE_K + tx;
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE_M; i += blockDim.y) {
            int a_global_row = blockIdx.y * BLOCK_SIZE_M + i + ty;
            if (a_global_row < M && a_global_col < K) {
                As[i + ty][tx] = A[a_global_row * K + a_global_col];
            } else {
                As[i + ty][tx] = 0.0f;
            }
        }
        
        // Load tile from B into shared memory (coalesced access)
        int b_global_row = t * BLOCK_SIZE_K + ty;
        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE_N; j += blockDim.x) {
            int b_global_col = blockIdx.x * BLOCK_SIZE_N + j + lane_id;
            if (b_global_row < K && b_global_col < N) {
                Bs[ty][j + lane_id] = B[b_global_row * N + b_global_col];
            } else {
                Bs[ty][j + lane_id] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute using shared memory tiles
        // Each thread computes a 4x4 block of C
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; k++) {
            // Load 4 elements from As for this thread
            float a_vals[4];
            int a_row = ty * 4;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                a_vals[i] = As[a_row + i][k];
            }
            
            // Load 4 elements from Bs for this thread
            float b_vals[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int b_col = lane_id * 4 + j;
                b_vals[j] = Bs[k][b_col];
            }
            
            // Compute 4x4 tile
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    c_reg[i * 4 + j] += a_vals[i] * b_vals[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results to global memory (coalesced access)
    if (row_start < M && col_start < N) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int row = row_start + i;
                int col = col_start + j;
                if (row < M && col < N) {
                    C[row * N + col] = c_reg[i * 4 + j];
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
    // Using 32x32 threads per block, each thread computes 4x4 tile
    dim3 block_dim(32, 32);  // 1024 threads per block (maximum)
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
    cpp_sources=matmul_hip_source_v2,
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