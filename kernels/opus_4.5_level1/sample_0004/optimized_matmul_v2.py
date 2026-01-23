import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

matmul_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Use larger tiles and register blocking for better performance
#define BLOCK_M 128
#define BLOCK_N 128  
#define BLOCK_K 16
#define THREAD_M 8
#define THREAD_N 8

__global__ void matmul_optimized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    // Thread block configuration: (BLOCK_N/THREAD_N) x (BLOCK_M/THREAD_M) = 16x16 = 256 threads
    __shared__ float As[BLOCK_K][BLOCK_M];
    __shared__ float Bs[BLOCK_K][BLOCK_N];
    
    int tx = threadIdx.x;  // 0-15, for N dimension
    int ty = threadIdx.y;  // 0-15, for M dimension
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Each thread computes THREAD_M x THREAD_N elements
    float accum[THREAD_M][THREAD_N] = {0.0f};
    
    // Starting position for this thread's output tile
    int row_start = by * BLOCK_M + ty * THREAD_M;
    int col_start = bx * BLOCK_N + tx * THREAD_N;
    
    int numTiles = (K + BLOCK_K - 1) / BLOCK_K;
    
    for (int t = 0; t < numTiles; t++) {
        // Load A tile (BLOCK_M x BLOCK_K) - each thread loads multiple elements
        // We have 256 threads, need to load BLOCK_M * BLOCK_K = 128 * 16 = 2048 elements
        // Each thread loads 8 elements
        int tid = ty * 16 + tx;  // 0-255
        
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = tid + i * 256;
            int a_row = by * BLOCK_M + (idx / BLOCK_K);
            int a_col = t * BLOCK_K + (idx % BLOCK_K);
            int sm_row = idx % BLOCK_K;
            int sm_col = idx / BLOCK_K;
            
            if (a_row < M && a_col < K) {
                As[sm_row][sm_col] = A[a_row * K + a_col];
            } else {
                As[sm_row][sm_col] = 0.0f;
            }
        }
        
        // Load B tile (BLOCK_K x BLOCK_N) - each thread loads multiple elements
        // Need to load BLOCK_K * BLOCK_N = 16 * 128 = 2048 elements
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = tid + i * 256;
            int b_row = t * BLOCK_K + (idx / BLOCK_N);
            int b_col = bx * BLOCK_N + (idx % BLOCK_N);
            int sm_row = idx / BLOCK_N;
            int sm_col = idx % BLOCK_N;
            
            if (b_row < K && b_col < N) {
                Bs[sm_row][sm_col] = B[b_row * N + b_col];
            } else {
                Bs[sm_row][sm_col] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute partial result
        #pragma unroll
        for (int k = 0; k < BLOCK_K; k++) {
            float a_vals[THREAD_M];
            float b_vals[THREAD_N];
            
            #pragma unroll
            for (int m = 0; m < THREAD_M; m++) {
                a_vals[m] = As[k][ty * THREAD_M + m];
            }
            
            #pragma unroll
            for (int n = 0; n < THREAD_N; n++) {
                b_vals[n] = Bs[k][tx * THREAD_N + n];
            }
            
            #pragma unroll
            for (int m = 0; m < THREAD_M; m++) {
                #pragma unroll
                for (int n = 0; n < THREAD_N; n++) {
                    accum[m][n] += a_vals[m] * b_vals[n];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results
    #pragma unroll
    for (int m = 0; m < THREAD_M; m++) {
        #pragma unroll
        for (int n = 0; n < THREAD_N; n++) {
            int out_row = row_start + m;
            int out_col = col_start + n;
            if (out_row < M && out_col < N) {
                C[out_row * N + out_col] = accum[m][n];
            }
        }
    }
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::empty({M, N}, A.options());
    
    dim3 blockDim(16, 16);  // 256 threads per block
    dim3 gridDim((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    
    matmul_optimized_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );
    
    return C;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B);
"""

matmul_module = load_inline(
    name="matmul_hip",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_hip_source,
    functions=["matmul_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_module

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.matmul_hip(A, B)


def get_inputs():
    M = 8205
    K = 2949
    N = 5921
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]


def get_init_inputs():
    return []
