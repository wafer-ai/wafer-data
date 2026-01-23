import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Define constants for tiled matrix multiplication
BLOCK_SIZE_M = 64
BLOCK_SIZE_N = 64
BLOCK_SIZE_K = 16
THREAD_M = 4
THREAD_N = 4

kernel_code = f"""
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define BLOCK_SIZE_M {BLOCK_SIZE_M}
#define BLOCK_SIZE_N {BLOCK_SIZE_N}
#define BLOCK_SIZE_K {BLOCK_SIZE_K}
#define THREAD_M {THREAD_M}
#define THREAD_N {THREAD_N}

__global__ void batched_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int m,
    int k,
    int n) {{
    
    // Batch index
    int batch_idx = blockIdx.z;
    
    // Matrix offsets for current batch
    const float* A_batch = A + batch_idx * m * k;
    const float* B_batch = B + batch_idx * k * n;
    float* C_batch = C + batch_idx * m * n;
    
    // Block position in output matrix
    int block_m = blockIdx.x * BLOCK_SIZE_M;
    int block_n = blockIdx.y * BLOCK_SIZE_N;
    
    // Thread position within the block
    int thread_idx_m = threadIdx.x / (BLOCK_SIZE_N / THREAD_N);
    int thread_idx_n = threadIdx.x % (BLOCK_SIZE_N / THREAD_N);
    
    // Allocate shared memory for tiles
    __shared__ float shared_A[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float shared_B[BLOCK_SIZE_K][BLOCK_SIZE_N];
    
    // Initialize accumulators for this thread
    float acc[THREAD_M][THREAD_N] = {{0.0f}};
    
    // Loop over k dimension in tiles
    for (int tile_k = 0; tile_k < k; tile_k += BLOCK_SIZE_K) {{
        // Load tile from A matrix into shared memory
        // Each thread loads THREAD_M * THREAD_N elements
        #pragma unroll
        for (int i = 0; i < THREAD_M; i++) {{
            #pragma unroll
            for (int j = 0; j < THREAD_N; j++) {{
                int global_m = block_m + thread_idx_m * THREAD_M + i;
                int global_k = tile_k + thread_idx_n * THREAD_N + j;
                
                if (global_m < m && global_k < k) {{
                    shared_A[thread_idx_m * THREAD_M + i][thread_idx_n * THREAD_N + j] = 
                        A_batch[global_m * k + global_k];
                }} else {{
                    shared_A[thread_idx_m * THREAD_M + i][thread_idx_n * THREAD_N + j] = 0.0f;
                }}
            }}
        }}
        
        // Load tile from B matrix into shared memory
        // Each thread loads THREAD_M * THREAD_N elements
        #pragma unroll
        for (int i = 0; i < THREAD_M; i++) {{
            #pragma unroll
            for (int j = 0; j < THREAD_N; j++) {{
                int global_k = tile_k + thread_idx_m * THREAD_M + i;
                int global_n = block_n + thread_idx_n * THREAD_N + j;
                
                if (global_k < k && global_n < n) {{
                    shared_B[thread_idx_m * THREAD_M + i][thread_idx_n * THREAD_N + j] = 
                        B_batch[global_k * n + global_n];
                }} else {{
                    shared_B[thread_idx_m * THREAD_M + i][thread_idx_n * THREAD_N + j] = 0.0f;
                }}
            }}
        }}
        
        __syncthreads();
        
        // Compute partial products
        #pragma unroll
        for (int tk = 0; tk < BLOCK_SIZE_K; tk++) {{
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) {{
                #pragma unroll
                for (int j = 0; j < THREAD_N; j++) {{
                    acc[i][j] += shared_A[thread_idx_m * THREAD_M + i][tk] * 
                                 shared_B[tk][thread_idx_n * THREAD_N + j];
                }}
            }}
        }}
        
        __syncthreads();
    }}
    
    // Write results to global memory
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {{
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {{
            int global_m = block_m + thread_idx_m * THREAD_M + i;
            int global_n = block_n + thread_idx_n * THREAD_N + j;
            
            if (global_m < m && global_n < n) {{
                C_batch[global_m * n + global_n] = acc[i][j];
            }}
        }}
    }}
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
    
    // Calculate grid dimensions
    int grid_m = (m + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;
    int grid_n = (n + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;
    
    dim3 grid(grid_m, grid_n, batch_size);
    dim3 block((BLOCK_SIZE_M / THREAD_M) * (BLOCK_SIZE_N / THREAD_N));
    
    // Launch kernel
    batched_gemm_kernel<<<grid, block>>>(
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
    Optimized batched matrix multiplication using custom tiled HIP kernel.
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
