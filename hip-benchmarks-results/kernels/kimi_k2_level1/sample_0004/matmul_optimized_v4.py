import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized matrix multiplication with improved memory access patterns
# Fixed correctness issues
matmul_hip_source_v4 = """
#include <hip/hip_runtime.h>

#define BM 128  // Block size M
#define BN 128  // Block size N
#define BK 32   // Block size K
#define TM 4    // Thread tile size M
#define TN 4    // Thread tile size N

__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N) {
    
    // Thread and block indices
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    
    // Initialize shared memory
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    
    // Thread local accumulators
    float c_reg[TM][TN];  // Each thread computes a TMxTN tile
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            c_reg[i][j] = 0.0f;
        }
    }
    
    // Starting positions
    int row_start_a = bidy * BM + tidy;
    int col_start_b = bidx * BN + tidx;
    
    // Loop over tiles in K dimension
    for (int tile = 0; tile < (K + BK - 1) / BK; tile++) {
        int tile_k = tile * BK;
        
        // Load tile from A into shared memory (coalesced load)
        #pragma unroll
        for (int i = 0; i < BM; i += blockDim.y) {
            int row = bidy * BM + i + tidy;
            int col = tile_k + tidx;
            if (row < M && col < K) {
                As[i + tidy][tidx] = A[row * K + col];
            } else {
                As[i + tidy][tidx] = 0.0f;
            }
        }
        
        // Load tile from B into shared memory (coalesced load)
        #pragma unroll
        for (int j = 0; j < BN; j += blockDim.x) {
            int row = tile_k + tidy;
            int col = bidx * BN + j + tidx;
            if (row < K && col < N) {
                Bs[tidy][j + tidx] = B[row * N + col];
            } else {
                Bs[tidy][j + tidx] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute using shared memory tiles
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            // Load As row for this thread
            float a_vals[TM];
            int my_row_start = tidy * TM;
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                a_vals[i] = As[my_row_start + i][k];
            }
            
            // Load Bs column for this thread
            float b_vals[TN];
            int my_col_start = tidx * TN;
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                b_vals[j] = Bs[k][my_col_start + j];
            }
            
            // Compute: c_reg += a_vals * b_vals
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    c_reg[i][j] += a_vals[i] * b_vals[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results to global memory (coalesced store)
    int my_row_start = bidy * BM + tidy * TM;
    int my_col_start = bidx * BN + tidx * TN;
    
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int row = my_row_start + i;
        if (row < M) {
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                int col = my_col_start + j;
                if (col < N) {
                    C[row * N + col] = c_reg[i][j];
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
    
    // Calculate grid and block sizes
    dim3 block_dim(32, 32);  // 1024 threads per block
    dim3 grid_dim(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );
    
    // Launch kernel
    hipLaunchKernelGGL(matmul_kernel, grid_dim, block_dim, 0, 0,
                       A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
                       M, K, N);
    
    return C;
}
"""

matmul_hip = load_inline(
    name="matmul_hip",
    cpp_sources=matmul_hip_source_v4,
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