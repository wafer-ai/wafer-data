import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

bmm_hip_source = """
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define TILE_SIZE 32

inline __device__ void load_tile_float4(float* __restrict__ dst, const float* __restrict__ src, int row, int col, int N, float pad) {
    if (col < N) {
        *dst = src[row * N + col];
    } else {
        *dst = pad;
    }
}

__global__ void batched_matmul_vectorized_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int batch_size, int M, int K, int N
) {
    // Thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int batch = blockIdx.z;
    
    // Calculate global position in output matrix
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    // Shared memory for tiles (128KB shared memory on MI300)
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Accumulator - use float
    float sum = 0.0f;
    
    // Calculate batch offset once
    int batch_offset_A = batch * M * K;
    int batch_offset_B = batch * K * N;
    int batch_offset_C = batch * M * N;
    
    // Loop over tiles in K dimension
    for (int t = 0; t < K; t += TILE_SIZE) {
        // Load tile of A with coalesced pattern
        int a_row = row;
        int a_col = t + tx;
        float a_val = 0.0f;
        if (a_row < M && a_col < K) {
            a_val = A[batch_offset_A + a_row * K + a_col];
        }
        As[ty][tx] = a_val;
        
        // Load tile of B with coalesced pattern
        int b_row = t + ty;
        int b_col = col;
        float b_val = 0.0f;
        if (b_row < K && b_col < N) {
            b_val = B[batch_offset_B + b_row * N + b_col];
        }
        Bs[ty][tx] = b_val;
        
        // Synchronize to ensure tiles are loaded
        __syncthreads();
        
        // Compute partial sum for this tile - unroll for efficiency
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[batch_offset_C + row * N + col] = sum;
    }
}

torch::Tensor batched_bmm_hip(torch::Tensor A, torch::Tensor B) {
    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);
    
    // Create output tensor with correct shape: (batch_size, M, N)
    auto C = torch::zeros({batch_size, M, N}, A.options());
    
    // Choose thread block size - maximum 1024 threads per block
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, batch_size);
    
    batched_matmul_vectorized_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );
    
    return C;
}
"""

# Load the inline HIP extension
batched_bmm = load_inline(
    name="batched_bmm",
    cpp_sources=bmm_hip_source,
    functions=["batched_bmm_hip"],
    verbose=True,
    with_cuda=True,
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized batched matrix multiplication using custom HIP kernel with shared memory tiling.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.batched_bmm = batched_bmm

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication using optimized HIP kernel.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return self.batched_bmm.batched_bmm_hip(A, B)