import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define TILE_SIZE 32

__global__ void tall_skinny_matmul_kernel(
    const float* __restrict__ A,  // M x K
    const float* __restrict__ B,  // K x M
    float* __restrict__ C,        // M x M
    int M, int K
) {
    // Shared memory with padding to avoid bank conflicts
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Since K is small (32), we only need one iteration for the K dimension
    // Load A tile: each thread loads one element
    if (row < M && threadIdx.x < K) {
        As[threadIdx.y][threadIdx.x] = A[row * K + threadIdx.x];
    } else {
        As[threadIdx.y][threadIdx.x] = 0.0f;
    }
    
    // Load B tile: each thread loads one element
    if (threadIdx.y < K && col < M) {
        Bs[threadIdx.y][threadIdx.x] = B[threadIdx.y * M + col];
    } else {
        Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }
    
    __syncthreads();
    
    // Compute dot product - unroll for K=32
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
        sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    
    // Write result
    if (row < M && col < M) {
        C[row * M + col] = sum;
    }
}

torch::Tensor tall_skinny_matmul_hip(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int M2 = B.size(1);
    
    auto C = torch::empty({M, M2}, A.options());
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((M2 + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    tall_skinny_matmul_kernel<<<grid, block>>>(
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
    extra_cuda_cflags=["-O3", "-ffast-math"]
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
