import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized kernel configuration
kernel_code = """
#include <hip/hip_runtime.h>

#define TILE_SIZE 32
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define UNROLL_FACTOR 4

__global__ void batched_gemm_optimized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int m,
    int k,
    int n) {
    
    // Batch index
    int batch_idx = blockIdx.z;
    
    // Matrix offsets
    const float* A_batch = A + batch_idx * m * k;
    const float* B_batch = B + batch_idx * k * n;
    float* C_batch = C + batch_idx * m * n;
    
    // Block position
    int block_row = blockIdx.y * BLOCK_SIZE_Y;
    int block_col = blockIdx.x * BLOCK_SIZE_X;
    
    // Thread indices
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    
    // Shared memory
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];
    
    // Register array for accumulation (using unrolling)
    float acc[UNROLL_FACTOR] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // Main loop over k
    for (int tile_k = 0; tile_k < k; tile_k += TILE_SIZE) {
        // Load A tile - each thread loads one element
        int A_row = block_row + ty;
        int A_col = tile_k + tx;
        
        for (int i = 0; i < BLOCK_SIZE_Y; i += 4) {
            int load_row = A_row + i;
            if (load_row < m && A_col < k) {
                A_tile[ty + i][tx] = A_batch[load_row * k + A_col];
            } else {
                A_tile[ty + i][tx] = 0.0f;
            }
        }
        
        // Load B tile - each thread loads one element
        int B_row = tile_k + ty;
        int B_col = block_col + tx;
        
        for (int i = 0; i < BLOCK_SIZE_X; i += 4) {
            int load_col = B_col + i;
            if (B_row < k && load_col < n) {
                B_tile[ty][tx + i] = B_batch[B_row * n + load_col];
            } else {
                B_tile[ty][tx + i] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute partial products using unrolling
        for (int k_idx = 0; k_idx < TILE_SIZE; k_idx++) {
            // Preload values for better instruction level parallelism
            float a_vals[UNROLL_FACTOR];
            for (int i = 0; i < UNROLL_FACTOR; i++) {
                int row = ty + i * 4;
                if (row < TILE_SIZE) {
                    a_vals[i] = A_tile[row][k_idx];
                } else {
                    a_vals[i] = 0.0f;
                }
            }
            
            float b_val = B_tile[k_idx][tx];
            
            // Accumulate
            for (int i = 0; i < UNROLL_FACTOR; i++) {
                acc[i] += a_vals[i] * b_val;
            }
        }
        
        __syncthreads();
    }
    
    // Write results with bounds checking
    int out_row = block_row + ty;
    int out_col = block_col + tx;
    
    if (out_row < m && out_col < n) {
        float sum = acc[0];
        for (int i = 1; i < UNROLL_FACTOR; i++) {
            sum += acc[i];
        }
        C_batch[out_row * n + out_col] = sum;
    }
}

torch::Tensor batched_gemm_hip(torch::Tensor A, torch::Tensor B) {
    // Check inputs
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3, "Expected 3D tensors");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions must match for matrix multiplication");
    
    int batch_size = A.size(0);
    int m = A.size(1);
    int k = A.size(2);
    int n = B.size(2);
    
    // Allocate output tensor
    auto C = torch::zeros({batch_size, m, n}, A.options());
    
    // Calculate grid dimensions
    dim3 threads_per_block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 num_blocks(
        (n + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (m + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
        batch_size
    );
    
    // Launch kernel
    batched_gemm_optimized_kernel<<<num_blocks, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size,
        m,
        k,
        n
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
    Optimized batched matrix multiplication using custom tiled HIP kernel with shared memory.
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
