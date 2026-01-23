import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized HIP kernel for MI300X GPU with register blocking and improved memory access
matmul_cpp_source = """
#include <hip/hip_runtime.h>

#define BM 64  // Block size for M dimension
#define BN 64  // Block size for N dimension  
#define BK 8   // Block size for K dimension
#define TM 4   // Thread tile size for M
#define TN 4   // Thread tile size for N

__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N) {
    
    // Shared memory tiles
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    
    // Register tiles for accumulating results
    float c_val[TM][TN];
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            c_val[i][j] = 0.0f;
        }
    }
    
    // Global positions
    int block_offset_m = blockIdx.y * BM;
    int block_offset_n = blockIdx.x * BN;
    
    int num_k_tiles = (N + BK - 1) / BK;
    
    // Loop over K tiles
    for (int tile_k = 0; tile_k < num_k_tiles; ++tile_k) {
        // Load A tile into shared memory (coalesced memory access)
        for (int i = 0; i < BM; i += blockDim.y) {
            int row = threadIdx.y * TM + i;
            int global_row = block_offset_m + row;
            int global_col = tile_k * BK + threadIdx.x;
            
            if (global_row < N && global_col < N) {
                As[row][threadIdx.x] = A[global_row * N + global_col];
            } else {
                As[row][threadIdx.x] = 0.0f;
            }
        }
        
        // Load B tile into shared memory (coalesced memory access)
        for (int j = 0; j < BN; j += blockDim.x) {
            int col = threadIdx.x * TN + j;
            int global_row = tile_k * BK + threadIdx.y;
            int global_col = block_offset_n + col;
            
            if (global_row < N && global_col < N) {
                Bs[threadIdx.y][col] = B[global_row * N + global_col];
            } else {
                Bs[threadIdx.y][col] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute dot product for this tile
        for (int k = 0; k < BK; ++k) {
            // Load A and B fragments into registers
            float a_frag[TM];
            for (int i = 0; i < TM; ++i) {
                a_frag[i] = As[threadIdx.y * TM + i][k];
            }
            
            float b_frag[TN];
            for (int j = 0; j < TN; ++j) {
                b_frag[j] = Bs[k][threadIdx.x * TN + j];
            }
            
            // Accumulate
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    c_val[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results to global memory (coalesced stores)
    for (int i = 0; i < TM; ++i) {
        int global_row = block_offset_m + threadIdx.y * TM + i;
        if (global_row < N) {
            for (int j = 0; j < TN; ++j) {
                int global_col = block_offset_n + threadIdx.x * TN + j;
                if (global_col < N) {
                    C[global_row * N + global_col] = c_val[i][j];
                }
            }
        }
    }
}

torch::Tensor matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);
    
    // Grid and block dimensions
    dim3 block_dim(BN / TN, BM / TM);
    dim3 grid_dim((N + BN - 1) / BN, (N + BM - 1) / BM);
    
    // Launch kernel
    hipLaunchKernelGGL(matmul_kernel, grid_dim, block_dim, 0, 0,
                       A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    
    return C;
}
"""

matmul_hip = load_inline(
    name="matmul_hip",
    cpp_sources=matmul_cpp_source,
    functions=["matmul"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_hip

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Move tensors to GPU if not already there
        if not A.is_cuda:
            A = A.cuda()
        if not B.is_cuda:
            B = B.cuda()
        
        # Ensure tensors are contiguous
        A = A.contiguous()
        B = B.contiguous()
        
        # Call the custom HIP kernel
        return self.matmul.matmul(A, B)

# Model configuration
N = 2048 * 2

def get_inputs():
    A = torch.rand(N, N).cuda()
    B = torch.rand(N, N).cuda()
    return [A, B]

def get_init_inputs():
    return []