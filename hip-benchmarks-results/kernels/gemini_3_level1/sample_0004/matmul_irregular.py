import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

matmul_source = """
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>

#define BM 64
#define BN 64
#define BK 32
#define TM 4
#define TN 4

__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;

    // 256 threads.
    // ty: 0..15, tx: 0..15
    int ty = tid / 16;
    int tx = tid % 16;

    // Shared Memory
    // As: Transposed Tile A (BK x BM) -> 32 x 64
    // Bs: Normal Tile B (BK x BN) -> 32 x 64
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    // Registers
    float c_reg[TM][TN];
    float a_reg[TM];
    float b_reg[TN];

    // Initialize accumulators
    #pragma unroll
    for(int i=0; i<TM; ++i) {
        #pragma unroll
        for(int j=0; j<TN; ++j) {
            c_reg[i][j] = 0.0f;
        }
    }

    int num_tiles = (K + BK - 1) / BK;

    for (int t = 0; t < num_tiles; ++t) {
        
        // Load A -> As (Transposed)
        // Tile BM x BK = 64 x 32 = 2048 floats
        // 256 threads -> 8 floats per thread
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int idx = tid + i * 256;
            int r = idx / BK; // 0..63
            int c = idx % BK; // 0..31
            
            int global_r = by * BM + r;
            int global_c = t * BK + c;
            
            float val = 0.0f;
            if (global_r < M && global_c < K) {
                val = A[global_r * K + global_c];
            }
            As[c][r] = val; 
        }

        // Load B -> Bs
        // Tile BK x BN = 32 x 64 = 2048 floats
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int idx = tid + i * 256;
            int r = idx / BN; // 0..31
            int c = idx % BN; // 0..63
            
            int global_r = t * BK + r;
            int global_c = bx * BN + c;
            
            float val = 0.0f;
            if (global_r < K && global_c < N) {
                val = B[global_r * N + global_c];
            }
            Bs[r][c] = val;
        }
        
        __syncthreads();
        
        // Compute
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // Vector load A from shared
            // Load 4 floats: As[k][ty*TM ... ty*TM+3]
            *reinterpret_cast<float4*>(&a_reg[0]) = *reinterpret_cast<float4*>(&As[k][ty * TM]);
            
            // Vector load B from shared
            // Load 4 floats: Bs[k][tx*TN ... tx*TN+3]
            *reinterpret_cast<float4*>(&b_reg[0]) = *reinterpret_cast<float4*>(&Bs[k][tx * TN]);
            
            // Outer product
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    c_reg[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store C
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int global_r = by * BM + ty * TM + i;
        if (global_r < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int global_c = bx * BN + tx * TN + j;
                if (global_c < N) {
                    C[global_r * N + global_c] = c_reg[i][j];
                }
            }
        }
    }
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    dim3 block(256);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}
"""

matmul_op = load_inline(
    name="matmul_irregular",
    cpp_sources=matmul_source,
    functions=["matmul_hip"],
    verbose=True,
    extra_cflags=["-O3", "-ffast-math"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_op = matmul_op
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_op.matmul_hip(A, B)
