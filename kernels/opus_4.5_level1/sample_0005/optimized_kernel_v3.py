import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Optimized for tall-skinny matrices where K (inner dimension) is small (32)
// Using larger tiles, vectorized loads, and register tiling

#define BLOCK_M 128
#define BLOCK_N 128
#define THREAD_M 8
#define THREAD_N 8
#define K_MAX 32

__global__ void tall_skinny_matmul_kernel_v3(
    const float* __restrict__ A,  // M x K
    const float* __restrict__ B,  // K x M
    float* __restrict__ C,        // M x M
    int M, int K
) {
    // Shared memory for A and B tiles
    __shared__ float As[BLOCK_M][K_MAX + 4];  // K is at most 32, padding for bank conflicts
    __shared__ float Bs[K_MAX][BLOCK_N + 4];
    
    int block_row = blockIdx.y * BLOCK_M;
    int block_col = blockIdx.x * BLOCK_N;
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;  // 256 threads
    
    // Initialize accumulators in registers
    float acc[THREAD_M][THREAD_N];
    #pragma unroll
    for (int m = 0; m < THREAD_M; ++m) {
        #pragma unroll
        for (int n = 0; n < THREAD_N; ++n) {
            acc[m][n] = 0.0f;
        }
    }
    
    // Load A tile into shared memory using vectorized loads where possible
    // A is M x K, we need BLOCK_M rows, K columns
    int total_A = BLOCK_M * K;
    for (int i = tid; i < total_A; i += num_threads) {
        int local_row = i / K;
        int local_col = i % K;
        int global_row = block_row + local_row;
        if (global_row < M) {
            As[local_row][local_col] = A[global_row * K + local_col];
        } else {
            As[local_row][local_col] = 0.0f;
        }
    }
    
    // Load B tile into shared memory
    // B is K x M, we need K rows, BLOCK_N columns
    int total_B = K * BLOCK_N;
    for (int i = tid; i < total_B; i += num_threads) {
        int local_row = i / BLOCK_N;
        int local_col = i % BLOCK_N;
        int global_col = block_col + local_col;
        if (global_col < M && local_row < K) {
            Bs[local_row][local_col] = B[local_row * M + global_col];
        } else {
            Bs[local_row][local_col] = 0.0f;
        }
    }
    
    __syncthreads();
    
    // Each thread computes THREAD_M x THREAD_N output elements
    int thread_row = threadIdx.y * THREAD_M;
    int thread_col = threadIdx.x * THREAD_N;
    
    // Compute - unroll the K loop
    #pragma unroll
    for (int k = 0; k < K_MAX; ++k) {
        if (k < K) {
            float a_vals[THREAD_M];
            float b_vals[THREAD_N];
            
            // Load A values into registers
            #pragma unroll
            for (int m = 0; m < THREAD_M; ++m) {
                a_vals[m] = As[thread_row + m][k];
            }
            
            // Load B values into registers
            #pragma unroll
            for (int n = 0; n < THREAD_N; ++n) {
                b_vals[n] = Bs[k][thread_col + n];
            }
            
            // Outer product
            #pragma unroll
            for (int m = 0; m < THREAD_M; ++m) {
                #pragma unroll
                for (int n = 0; n < THREAD_N; ++n) {
                    acc[m][n] = __fmaf_rn(a_vals[m], b_vals[n], acc[m][n]);
                }
            }
        }
    }
    
    // Write results - using vectorized stores where possible
    #pragma unroll
    for (int m = 0; m < THREAD_M; ++m) {
        int global_row = block_row + thread_row + m;
        if (global_row < M) {
            #pragma unroll
            for (int n = 0; n < THREAD_N; ++n) {
                int global_col = block_col + thread_col + n;
                if (global_col < M) {
                    C[global_row * M + global_col] = acc[m][n];
                }
            }
        }
    }
}

torch::Tensor tall_skinny_matmul_hip(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int M2 = B.size(1);
    
    auto C = torch::empty({M, M2}, A.options());
    
    // BLOCK_M / THREAD_M threads in y, BLOCK_N / THREAD_N threads in x
    dim3 block(BLOCK_N / THREAD_N, BLOCK_M / THREAD_M);  // 16 x 16 = 256 threads
    dim3 grid((M2 + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    
    tall_skinny_matmul_kernel_v3<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K
    );
    
    return C;
}
"""

cpp_source = """
torch::Tensor tall_skinny_matmul_hip(torch::Tensor A, torch::Tensor B);
"""

tall_skinny_matmul = load_inline(
    name="tall_skinny_matmul",
    cpp_sources=cpp_source,
    cuda_sources=hip_source,
    functions=["tall_skinny_matmul_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math", "-munsafe-fp-atomics"]
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tall_skinny_matmul = tall_skinny_matmul
    
    def forward(self, A, B):
        return self.tall_skinny_matmul.tall_skinny_matmul_hip(A, B)


def get_inputs():
    M = 16384 * 2
    N = 16 * 2
    A = torch.rand(M, N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]


def get_init_inputs():
    return []
