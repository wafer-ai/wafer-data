
import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

gemm_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void tall_skinny_gemm_kernel_v2(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    // Each thread computes a 8x8 tile of C.
    // Each block is 16x16 threads -> 128x128 tile of C.
    
    const int TILE_SIZE = 128;
    int block_row = blockIdx.y * TILE_SIZE;
    int block_col = blockIdx.x * TILE_SIZE;
    
    int tx = threadIdx.x; // 0..15
    int ty = threadIdx.y; // 0..15
    
    // Shared memory for A (128x32) and B (32x128)
    __shared__ float s_A[128][33]; // 128*33 to avoid bank conflicts
    __shared__ float s_B[32][129]; // 32*129 to avoid bank conflicts
    
    // Load A into shared memory (128x32 = 4096 elements).
    // 256 threads, each loads 16 elements.
    for (int i = 0; i < 16; ++i) {
        int idx = (ty * 16 + tx) * 16 + i;
        int r = idx / 32;
        int c = idx % 32;
        if (block_row + r < M && c < K) {
            s_A[r][c] = A[(block_row + r) * K + c];
        } else {
            s_A[r][c] = 0.0f;
        }
    }
    
    // Load B into shared memory (32x128 = 4096 elements).
    // 256 threads, each loads 16 elements.
    for (int i = 0; i < 16; ++i) {
        int idx = (ty * 16 + tx) * 16 + i;
        int r = idx / 128;
        int c = idx % 128;
        if (r < K && block_col + c < N) {
            s_B[r][c] = B[r * N + (block_col + c)];
        } else {
            s_B[r][c] = 0.0f;
        }
    }
    
    __syncthreads();
    
    float res[8][8] = {0.0f};
    
    #pragma unroll
    for (int k = 0; k < 32; ++k) {
        float a_vals[8];
        float b_vals[8];
        
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            a_vals[i] = s_A[ty * 8 + i][k];
        }
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            b_vals[j] = s_B[k][tx * 8 + j];
        }
        
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                res[i][j] += a_vals[i] * b_vals[j];
            }
        }
    }
    
    // Store 8x8 results to C using float4 if possible.
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int r = block_row + ty * 8 + i;
        if (r < M) {
            #pragma unroll
            for (int j = 0; j < 8; j += 4) {
                int c = block_col + tx * 8 + j;
                if (c + 3 < N) {
                    float4 val;
                    val.x = res[i][j];
                    val.y = res[i][j+1];
                    val.z = res[i][j+2];
                    val.w = res[i][j+3];
                    *((float4*)(&C[r * N + c])) = val;
                } else {
                    for (int jj = 0; jj < 4; ++jj) {
                        if (c + jj < N) {
                            C[r * N + c + jj] = res[i][j + jj];
                        }
                    }
                }
            }
        }
    }
}

torch::Tensor tall_skinny_gemm_hip(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::empty({M, N}, A.options());
    
    dim3 block_dim(16, 16);
    dim3 grid_dim((N + 127) / 128, (M + 127) / 128);
    
    tall_skinny_gemm_kernel_v2<<<grid_dim, block_dim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    
    return C;
}
"""

tall_skinny_gemm = load_inline(
    name="tall_skinny_gemm_v2",
    cpp_sources=gemm_cpp_source,
    functions=["tall_skinny_gemm_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tall_skinny_gemm = tall_skinny_gemm
    
    def forward(self, A, B):
        return self.tall_skinny_gemm.tall_skinny_gemm_hip(A, B)
