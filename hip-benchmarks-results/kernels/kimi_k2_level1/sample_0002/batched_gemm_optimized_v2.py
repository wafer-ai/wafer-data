import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Define constants for tiled matrix multiplication
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 16

kernel_code = """
#include <hip/hip_runtime.h>

#define BLOCK_SIZE_M 128
#define BLOCK_SIZE_N 128
#define BLOCK_SIZE_K 16

__global__ void batched_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int m,
    int k,
    int n) {
    
    // Batch index
    int batch_idx = blockIdx.z;
    
    // Matrix offsets for current batch
    const float* A_batch = A + batch_idx * m * k;
    const float* B_batch = B + batch_idx * k * n;
    float* C_batch = C + batch_idx * m * n;
    
    // Block position in output matrix
    int block_m = blockIdx.x * BLOCK_SIZE_M;
    int block_n = blockIdx.y * BLOCK_SIZE_N;
    
    // Thread indices within the block
    int threadIdx_m = threadIdx.y;
    int threadIdx_n = threadIdx.x;
    
    // Shared memory for tiles
    __shared__ float shared_A[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float shared_B[BLOCK_SIZE_K][BLOCK_SIZE_N];
    
    // Register array for accumulating results
    float acc[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            acc[i][j] = 0.0f;
        }
    }
    
    // Loop over k-dimension tiles
    for (int tile_k = 0; tile_k < k; tile_k += BLOCK_SIZE_K) {
        // Load tile from A matrix into shared memory
        // Each thread loads a single element
        int load_m = block_m + threadIdx.y * 8 + threadIdx.x / 1;
        int load_k = tile_k + (threadIdx.x % 1);
        
        for (int i = 0; i < 8; i++) {
            int global_m = block_m + threadIdx.y * 8 + i;
            int global_k = tile_k + threadIdx.x;
            
            if (global_m < m && global_k < k && threadIdx.x < BLOCK_SIZE_K) {
                shared_A[threadIdx.y * 8 + i][threadIdx.x] = A_batch[global_m * k + global_k];
            } else if (threadIdx.x < BLOCK_SIZE_K) {
                shared_A[threadIdx.y * 8 + i][threadIdx.x] = 0.0f;
            }
        }
        
        // Load tile from B matrix into shared memory
        for (int i = 0; i < 8; i++) {
            int global_k = tile_k + threadIdx.y;
            int global_n = block_n + threadIdx.x * 8 + i;
            
            if (global_k < k && global_n < n) {
                shared_B[threadIdx.y][threadIdx.x * 8 + i] = B_batch[global_k * n + global_n];
            } else {
                shared_B[threadIdx.y][threadIdx.x * 8 + i] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute partial products
        for (int tk = 0; tk < BLOCK_SIZE_K; tk++) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                float a_val = shared_A[threadIdx.y * 8 + i][tk];
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    acc[i][j] += a_val * shared_B[tk][threadIdx.x * 8 + j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results to global memory
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int global_m = block_m + threadIdx.y * 8 + i;
        if (global_m >= m) continue;
        
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int global_n = block_n + threadIdx.x * 8 + j;
            if (global_n >= n) continue;
            
            C_batch[global_m * n + global_n] = acc[i][j];
        }
    }
}}

__global__ void simple_batched_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int m,
    int k,
    int n) {
    
    // Simpler implementation that should be correct
    int batch_idx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || row >= m || col >= n) return;
    
    // Matrix offsets for current batch
    const float* A_batch = A + batch_idx * m * k;
    const float* B_batch = B + batch_idx * k * n;
    float* C_batch = C + batch_idx * m * n;
    
    float sum = 0.0f;
    for (int k_idx = 0; k_idx < k; k_idx++) {
        sum += A_batch[row * k + k_idx] * B_batch[k_idx * n + col];
    }
    
    C_batch[row * n + col] = sum;
}}

torch::Tensor batched_gemm_hip(torch::Tensor A, torch::Tensor B) {{
    // Check inputs
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3, "Expected 3D tensors");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions must match for matrix multiplication");
    
    int batch_size = A.size(0);
    int m = A.size(1);
    int k = A.size(2);
    int n = B.size(2);
    
    // Allocate output tensor
    auto C = torch::zeros({{batch_size, m, n}}, A.options());
    
    // Calculate grid dimensions (simpler version)
    dim3 threads_per_block(16, 16);
    dim3 num_blocks((n + threads_per_block.x - 1) / threads_per_block.x,
                    (m + threads_per_block.y - 1) / threads_per_block.y,
                    batch_size);
    
    // Launch simple kernel first
    simple_batched_gemm_kernel<<<num_blocks, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size,
        m,
        k,
        n
    );
    
    return C;
}}
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
    Optimized batched matrix multiplication using custom HIP kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.batched_gemm = batched_gemm
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are contiguous and on CUDA
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
