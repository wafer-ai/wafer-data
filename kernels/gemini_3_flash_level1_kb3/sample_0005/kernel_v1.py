
import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

gemm_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void tall_skinny_gemm_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    // A: (M, K), B: (K, N), C: (M, N)
    // In this case, K is small (32).
    
    // Each block handles a TILE_M x TILE_N portion of C.
    // Let's use TILE_M = 64, TILE_N = 64.
    // Each thread handles a 4x4 portion of C.
    // Threads in block: 16x16.
    
    const int TILE_M = 64;
    const int TILE_N = 64;
    
    int block_row = blockIdx.y * TILE_M;
    int block_col = blockIdx.x * TILE_N;
    
    int tx = threadIdx.x; // 0..15
    int ty = threadIdx.y; // 0..15
    
    __shared__ float s_A[TILE_M][32]; // 64 * 32 * 4 = 8KB
    __shared__ float s_B[32][TILE_N]; // 32 * 64 * 4 = 8KB
    
    // Load A and B into shared memory
    // Each block has 256 threads.
    // s_A has 2048 elements. Each thread loads 8 elements.
    for (int i = 0; i < 8; ++i) {
        int idx = (ty * 16 + tx) * 8 + i;
        int r = idx / 32;
        int c = idx % 32;
        if (block_row + r < M && c < K) {
            s_A[r][c] = A[(block_row + r) * K + c];
        } else {
            s_A[r][c] = 0.0f;
        }
    }
    
    // s_B has 2048 elements. Each thread loads 8 elements.
    for (int i = 0; i < 8; ++i) {
        int idx = (ty * 16 + tx) * 8 + i;
        int r = idx / TILE_N;
        int c = idx % TILE_N;
        if (r < K && block_col + c < N) {
            s_B[r][c] = B[r * N + (block_col + c)];
        } else {
            s_B[r][c] = 0.0f;
        }
    }
    
    __syncthreads();
    
    // Each thread computes a 4x4 tile of C.
    float res[4][4] = {0.0f};
    
    for (int k = 0; k < 32; ++k) {
        float a_vals[4];
        float b_vals[4];
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            a_vals[i] = s_A[ty * 4 + i][k];
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            b_vals[j] = s_B[k][tx * 4 + j];
        }
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                res[i][j] += a_vals[i] * b_vals[j];
            }
        }
    }
    
    // Store res[4][4] back to C.
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            int r = block_row + ty * 4 + i;
            int c = block_col + tx * 4 + j;
            if (r < M && c < N) {
                C[r * N + c] = res[i][j];
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
    dim3 grid_dim((N + 63) / 64, (M + 63) / 64);
    
    tall_skinny_gemm_kernel<<<grid_dim, block_dim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    
    return C;
}
"""

tall_skinny_gemm = load_inline(
    name="tall_skinny_gemm",
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

