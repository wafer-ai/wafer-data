import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// For tall-skinny matmul where K is small (32)
// A is M x K, B is K x M, C is M x M
// Each output row of C can be computed by taking one row of A (K elements)
// and multiplying with all of B (K x M)

#define WARP_SIZE 64
#define NUM_WARPS 4
#define BLOCK_SIZE (WARP_SIZE * NUM_WARPS)
#define TILE_N 256

__global__ void tall_skinny_matmul_kernel_v4(
    const float* __restrict__ A,  // M x K
    const float* __restrict__ B,  // K x M
    float* __restrict__ C,        // M x M
    int M, int K
) {
    // Each block handles one row of output, tiled across columns
    int row = blockIdx.y;
    int tile_col_start = blockIdx.x * TILE_N;
    
    if (row >= M) return;
    
    // Load row of A into registers (K is small, ~32)
    float a_reg[32];
    for (int k = threadIdx.x; k < K; k += BLOCK_SIZE) {
        if (k < 32) a_reg[k] = A[row * K + k];
    }
    // Broadcast within the block using shared memory
    __shared__ float a_shared[32];
    
    for (int k = threadIdx.x; k < K; k += BLOCK_SIZE) {
        a_shared[k] = A[row * K + k];
    }
    __syncthreads();
    
    // Load into registers
    #pragma unroll
    for (int k = 0; k < 32; ++k) {
        if (k < K) a_reg[k] = a_shared[k];
        else a_reg[k] = 0.0f;
    }
    
    // Each thread computes multiple output elements
    for (int col = tile_col_start + threadIdx.x; col < min(tile_col_start + TILE_N, M); col += BLOCK_SIZE) {
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            if (k < K) {
                sum += a_reg[k] * B[k * M + col];
            }
        }
        C[row * M + col] = sum;
    }
}

torch::Tensor tall_skinny_matmul_hip(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int M2 = B.size(1);
    
    auto C = torch::empty({M, M2}, A.options());
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((M2 + TILE_N - 1) / TILE_N, M);
    
    tall_skinny_matmul_kernel_v4<<<grid, block>>>(
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
