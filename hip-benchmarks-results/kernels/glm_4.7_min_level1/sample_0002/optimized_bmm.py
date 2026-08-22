import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

batched_gemm_cpp_source = """
#include <hip/hip_runtime.h>

#define BLOCK_SIZE 32

__global__ void batched_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch, int m, int k, int n)
{
    int batch_idx = blockIdx.z;
    
    const float* __restrict__ A_batch = A + batch_idx * m * k;
    const float* __restrict__ B_batch = B + batch_idx * k * n;
    float* __restrict__ C_batch = C + batch_idx * m * n;
    
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    float sum = 0.0f;
    
    for (int kk = 0; kk < (k + BLOCK_SIZE - 1) / BLOCK_SIZE; ++kk) {
        int K_start = kk * BLOCK_SIZE;
        
        // Load one element of A and B per thread
        if (row < m && K_start + threadIdx.x < k) {
            As[threadIdx.y][threadIdx.x] = A_batch[row * k + K_start + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (K_start + threadIdx.y < k && col < n) {
            Bs[threadIdx.y][threadIdx.x] = B_batch[(K_start + threadIdx.y) * n + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial products
        for (int e = 0; e < BLOCK_SIZE; ++e) {
            sum += As[threadIdx.y][e] * Bs[e][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < m && col < n) {
        C_batch[row * n + col] = sum;
    }
}

torch::Tensor batched_gemm_hip(torch::Tensor A, torch::Tensor B) {
    int batch = A.size(0);
    int m = A.size(1);
    int k = A.size(2);
    int n = B.size(2);
    
    auto C = torch::zeros({batch, m, n}, A.options());
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE, batch);
    
    batched_gemm_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch, m, k, n
    );
    
    return C;
}
"""

batched_gemm = load_inline(
    name="batched_gemm",
    cpp_sources=batched_gemm_cpp_source,
    functions=["batched_gemm_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
    Uses an optimized HIP kernel with block tiling.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.batched_gemm = batched_gemm
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication using optimized HIP kernel.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        A = A.cuda().contiguous()
        B = B.cuda().contiguous()
        
        return self.batched_gemm.batched_gemm_hip(A, B)