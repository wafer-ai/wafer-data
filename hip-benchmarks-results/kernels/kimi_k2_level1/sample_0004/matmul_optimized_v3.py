import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized matrix multiplication for AMD MI300X
# Uses 2D blocktiling with vectorized loads and better memory layout
matmul_hip_source_v3 = """
#include <hip/hip_runtime.h>
#include <hip/hip_common.h>

// Configuration optimized for MI300X
#define BM 256   // Block size for M dimension
#define BN 256   // Block size for N dimension
#define BK 32    // Block size for K dimension
#define TM 4     // Thread tile size for M
#define TN 4     // Thread tile size for N

__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N) {
    
    // Thread indices
    int tidx = threadIdx.x;  // 0-255
    int tidy = threadIdx.y;  // 0-7
    
    // Warp and lane info (AMD uses 64 threads per wavefront)
    int warp_id = tidx >> 5;  // Warp ID (0-7)
    int lane_id = tidx & 31;  // Lane within warp (0-31)
    
    // Block global position
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    
    // Initialize shared memory
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    
    // Thread-local accumulators (registers)
    float acc[TM][TN] = {{0.0f}};
    float As_vals[TM];
    float Bs_vals[TN];
    
    // Calculate work per thread
    int num_tiles = (K + BK - 1) / BK;
    
    // Loop over K dimension
    for (int tile = 0; tile < num_tiles; tile++) {
        // Load A tile into shared memory (coalesced)
        int k_offset = tile * BK;
        
        // Each thread loads 4 elements from A
        for (int i = 0; i < BM; i += blockDim.y * 4) {
            int row = block_row + i + tidy;
            int col = k_offset + tidx;
            if (row < M && col < K) {
                As[i + tidy][tidx] = A[row * K + col];
            } else {
                As[i + tidy][tidx] = 0.0f;
            }
        }
        
        // Load B tile into shared memory (coalesced)
        for (int i = 0; i < BK; i += blockDim.y) {
            int row = k_offset + i + tidy;
            int col = block_col + tidx;
            if (row < K && col < N) {
                Bs[i + tidy][tidx] = B[row * N + col];
            } else {
                Bs[i + tidy][tidx] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute using shared memory
        for (int k = 0; k < BK; k++) {
            // Load As values - each thread needs TM elements from As
            int As_row = tidy * TM;
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                As_vals[i] = As[As_row + i][k];
            }
            
            // Load Bs values - each thread needs TN elements from Bs
            int Bs_col = tidx * TN;
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                Bs_vals[j] = Bs[k][Bs_col + j];
            }
            
            // Compute: acc += As * Bs
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    acc[i][j] += As_vals[i] * Bs_vals[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results back to global memory (coalesced)
    int base_row = block_row + tidy * TM;
    int base_col = block_col + tidx * TN;
    
    if (base_row < M && base_col < N) {
        #pragma unroll
        for (int i = 0; i < TM; i++) {
            int row = base_row + i;
            if (row < M) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    int col = base_col + j;
                    if (col < N) {
                        C[row * N + col] = acc[i][j];
                    }
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
    
    // Configure grid and block dimensions
    dim3 block_dim(64, 8);  // 512 threads per block
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
    cpp_sources=matmul_hip_source_v3,
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