import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

matmul_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Use larger tiles and register blocking for better performance
#define TILE_M 128
#define TILE_N 128
#define TILE_K 16
#define THREAD_M 8
#define THREAD_N 8

__global__ void matmul_optimized_kernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int M, int K, int N) {
    // Each block computes TILE_M x TILE_N of C
    // Each thread computes THREAD_M x THREAD_N elements
    
    __shared__ float As[TILE_K][TILE_M];
    __shared__ float Bs[TILE_K][TILE_N];
    
    // Thread indices within the block
    int tx = threadIdx.x;  // 0 to 15
    int ty = threadIdx.y;  // 0 to 15
    
    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Starting position for this block
    int rowStart = by * TILE_M;
    int colStart = bx * TILE_N;
    
    // Thread linear ID for loading
    int threadId = ty * blockDim.x + tx;
    int numThreads = blockDim.x * blockDim.y; // 256
    
    // Register accumulator array
    float acc[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            acc[i][j] = 0.0f;
        }
    }
    
    // Loop over K dimension in tiles
    for (int tileK = 0; tileK < K; tileK += TILE_K) {
        // Cooperative loading of A tile (TILE_M x TILE_K) into shared memory
        // A is M x K, we need A[rowStart:rowStart+TILE_M, tileK:tileK+TILE_K]
        // Store transposed for coalesced access later
        for (int idx = threadId; idx < TILE_M * TILE_K; idx += numThreads) {
            int m = idx % TILE_M;
            int k = idx / TILE_M;
            int globalRow = rowStart + m;
            int globalCol = tileK + k;
            if (globalRow < M && globalCol < K) {
                As[k][m] = A[globalRow * K + globalCol];
            } else {
                As[k][m] = 0.0f;
            }
        }
        
        // Cooperative loading of B tile (TILE_K x TILE_N) into shared memory
        // B is K x N, we need B[tileK:tileK+TILE_K, colStart:colStart+TILE_N]
        for (int idx = threadId; idx < TILE_K * TILE_N; idx += numThreads) {
            int k = idx / TILE_N;
            int n = idx % TILE_N;
            int globalRow = tileK + k;
            int globalCol = colStart + n;
            if (globalRow < K && globalCol < N) {
                Bs[k][n] = B[globalRow * N + globalCol];
            } else {
                Bs[k][n] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute partial results
        // Each thread handles THREAD_M x THREAD_N elements
        int threadRowStart = ty * THREAD_M;
        int threadColStart = tx * THREAD_N;
        
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            // Load a column of As and a row of Bs into registers
            float a_reg[THREAD_M];
            float b_reg[THREAD_N];
            
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) {
                a_reg[i] = As[k][threadRowStart + i];
            }
            
            #pragma unroll
            for (int j = 0; j < THREAD_N; j++) {
                b_reg[j] = Bs[k][threadColStart + j];
            }
            
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_N; j++) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results to global memory
    int threadRowStart = ty * THREAD_M;
    int threadColStart = tx * THREAD_N;
    
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            int globalRow = rowStart + threadRowStart + i;
            int globalCol = colStart + threadColStart + j;
            if (globalRow < M && globalCol < N) {
                C[globalRow * N + globalCol] = acc[i][j];
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
    
    TORCH_CHECK(B.size(0) == K, "Matrix dimensions mismatch");
    
    auto C = torch::zeros({M, N}, A.options());
    
    // 16x16 threads per block, each computing 8x8 elements = 128x128 per block
    dim3 blockDim(16, 16);
    dim3 gridDim((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    
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
