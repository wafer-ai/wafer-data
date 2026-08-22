import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

matmul_hip_source = """
#include <hip/hip_runtime.h>

#define TILE_SIZE 16

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; ++t) {
        // Cooperative load of A tile
        for (int load_offset = 0; load_offset < TILE_SIZE; load_offset += blockDim.x) {
            int lane = ty * blockDim.x + load_offset + tx;
            if (lane < TILE_SIZE) {
                int a_col = t * TILE_SIZE + lane;
                if (row < M && a_col < K) {
                    As[ty][lane] = A[row * K + a_col];
                } else {
                    As[ty][lane] = 0.0f;
                }
            }
        }
        
        // Cooperative load of B tile  
        for (int load_offset = 0; load_offset < TILE_SIZE; load_offset += blockDim.y) {
            int lane = tx * blockDim.y + load_offset + ty;
            if (lane < TILE_SIZE) {
                int b_row = t * TILE_SIZE + lane;
                if (b_row < K && col < N) {
                    Bs[lane][tx] = B[b_row * N + col];
                } else {
                    Bs[lane][tx] = 0.0f;
                }
            }
        }
        
        __syncthreads();
        
        // Compute dot product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    dim3 block(16, 16);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    hipLaunchKernelGGL(HIP_KERNEL_NAME(matmul_kernel), grid, block, 0, 0, 
                       A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    
    return C;
}
"""

matmul_module = load_inline(
    name="matmul_optimized",
    cpp_sources=matmul_hip_source,
    functions=["matmul_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul_module
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if not A.is_cuda:
            A = A.cuda()
        if not B.is_cuda:
            B = B.cuda()
        
        # Use FP32 precision
        A = A.float()
        B = B.float()
        
        return self.matmul.matmul_hip(A, B)