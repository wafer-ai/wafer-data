import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

matmul_cpp_source = """
#include <hip/hip_runtime.h>

#define TILE_SIZE 32

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int threadIdx1D = threadIdx.y * blockDim.x + threadIdx.x;
    
    // Use 1D indexing for better coalescing
    int tx = threadIdx1D % TILE_SIZE;
    int ty = threadIdx1D / TILE_SIZE;
    
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float acc = 0.0f;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; ++t) {
        // Load A tile - each thread loads one element
        int aCol = t * TILE_SIZE + tx;
        if (row < N && aCol < N) {
            As[ty][tx] = A[row * N + aCol];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load B tile - each thread loads one element
        int bRow = t * TILE_SIZE + ty;
        if (bRow < N && col < N) {
            Bs[ty][tx] = B[bRow * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < N && col < N) {
        C[row * N + col] = acc;
    }
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());
    
    dim3 block(32, 8); // 256 threads per block
    int gridX = (N + TILE_SIZE - 1) / TILE_SIZE;
    int gridY = (N + TILE_SIZE - 1) / TILE_SIZE;
    dim3 grid(gridX, gridY);
    
    matmul_kernel<<<grid, block, 0, 0>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        N
    );
    
    return C;
}
"""

matmul_module = load_inline(
    name="matmul_optimized",
    cpp_sources=matmul_cpp_source,
    functions=["matmul_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication using custom HIP kernel
    with shared memory tiling for better performance.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_module = matmul_module
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using optimized HIP kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return self.matmul_module.matmul_hip(A, B)


N = 2048 * 2

def get_inputs():
    A = torch.rand(N, N).cuda()
    B = torch.rand(N, N).cuda()
    return [A, B]

def get_init_inputs():
    return []