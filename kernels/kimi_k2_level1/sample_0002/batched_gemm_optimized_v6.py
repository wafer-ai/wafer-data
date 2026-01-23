import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized kernel configuration for MI300X
kernel_code = """
#include <hip/hip_runtime.h>

#define TILE_M 128
#define TILE_N 128
#define TILE_K 16
#define BLOCK_M 64
#define BLOCK_N 64
#define THREAD_M 8
#define THREAD_N 8

__global__ void batched_gemm_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int m, int k, int n) {
    
    // Batch index
    int batch_idx = blockIdx.z;
    
    // Matrix offsets
    const float* A_batch = A + batchIdx.z * m * k;
    const float* B_batch = B + batchIdx.z * k * n;
    float* C_batch = C + batchIdx.z * m * n;
    
    // Block position in M and N dimensions
    int block_m = blockIdx.x * TILE_M;
    int block_n = blockIdx.y * TILE_N;
    
    // Thread position within the tile
    int tid = threadIdx.x;
    int thread_m = tid / (TILE_N / THREAD_N);
    int thread_n = tid % (TILE_N / THREAD_N);
    
    // Shared memory for tiles
    __shared__ float A_tile[TILE_M][TILE_K];
    __shared__ float B_tile[TILE_K][TILE_N];
    
    // Accumulators
    float acc[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            acc[i][j] = 0.0f;
        }
    }
    
    // Main loop over K dimension
    for (int tile_k = 0; tile_k < k; tile_k += TILE_K) {
        // Load A tile from global memory (coalesced access)
        #pragma unroll
        for (int i = 0; i < THREAD_M; i++) {
            int global_m = block_m + thread_m * THREAD_M + i;
            int global_k = tile_k + thread_n;
            
            if (global_m < m && global_k < k) {
                A_tile[thread_m * THREAD_M + i][thread_n] = A_batch[global_m * k + global_k];
            } else {
                A_tile[thread_m * THREAD_M + i][thread_n] = 0.0f;
            }
        }
        
        // Load B tile from global memory (coalesced access)
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            int global_k = tile_k + thread_m;
            int global_n = block_n + thread_n * THREAD_N + j;
            
            if (global_k < k && global_n < n) {
                B_tile[thread_m][thread_n * THREAD_N + j] = B_batch[global_k * n + global_n];
            } else {
                B_tile[thread_m][thread_n * THREAD_N + j] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Multiply tiles and accumulate
        #pragma unroll
        for (int k_idx = 0; k_idx < TILE_K; k_idx++) {
            float a_vals[THREAD_M];
            float b_vals[THREAD_N];
            
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) {
                a_vals[i] = A_tile[thread_m * THREAD_M + i][k_idx];
            }
            
            #pragma unroll
            for (int j = 0; j < THREAD_N; j++) {
                b_vals[j] = B_tile[k_idx][thread_n * THREAD_N + j];
            }
            
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_N; j++) {
                    acc[i][j] += a_vals[i] * b_vals[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results to global memory (coalesced access)
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        int global_m = block_m + thread_m * THREAD_M + i;
        if (global_m >= m) continue;
        
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            int global_n = block_n + thread_n * THREAD_N + j;
            if (global_n >= n) continue;
            
            C_batch[global_m * n + global_n] = acc[i][j];
        }
    }
}

torch::Tensor batched_gemm_hip(torch::Tensor A, torch::Tensor B) {
    // Validate inputs
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3, "Input tensors must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch dimensions must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Incompatible matrix dimensions");
    
    int batch_size = A.size(0);
    int m = A.size(1);
    int k = A.size(2);
    int n = B.size(2);
    
    // Allocate output
    auto C = torch::zeros({batch_size, m, n}, A.options());
    
    // Launch parameters
    dim3 threads_per_block((TILE_N / THREAD_N) * (TILE_M / THREAD_M));
    dim3 num_blocks(
        (m + TILE_M - 1) / TILE_M,
        (n + TILE_N - 1) / TILE_N,
        batch_size
    );
    
    // Launch kernel
    batched_gemm_tiled_kernel<<<num_blocks, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, m, k, n
    );
    
    return C;
}
"""

# Compile the kernel
batched_gemm = load_inline(
    name="batched_gemm",
    cpp_sources=kernel_code,
    functions=["batched_gemm_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Highly optimized batched matrix multiplication for AMD MI300X.
    Uses tiled computation with shared memory for optimal performance.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.batched_gemm = batched_gemm
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.cuda().contiguous()
        B = B.cuda().contiguous()
        
        return self.batched_gemm.batched_gemm_hip(A, B)

def get_inputs():
    batch_size = 128
    m = 128 * 4
    k = 256 * 4
    n = 512 * 4
    
    A = torch.randn(batch_size, m, k, dtype=torch.float32, device='cuda')
    B = torch.randn(batch_size, k, n, dtype=torch.float32, device='cuda')
    return [A, B]

def get_init_inputs():
    return []
