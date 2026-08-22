import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

matvec_cpp_source = """
#include <hip/hip_runtime.h>

#define TILE_SIZE 512

__global__ void matvec_kernel(const float* A, const float* B, float* C, int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float B_shared[TILE_SIZE];
    float sum = 0.0f;
    
    for (int tile_start = 0; tile_start < K; tile_start += TILE_SIZE) {
        // Load B tile into shared memory
        int b_idx = threadIdx.x;
        if (tile_start + b_idx < K && b_idx < TILE_SIZE) {
            B_shared[b_idx] = B[tile_start + b_idx];
        }
        __syncthreads();
        
        // Unroll loop for better performance
        if (row < M) {
            const float* A_row = &A[row * K + tile_start];
            int elements = min(TILE_SIZE, K - tile_start);
            
            // Process 4 elements per iteration (vectorized loads)
            int i = 0;
            for (; i + 4 <= elements; i += 4) {
                sum += A_row[i] * B_shared[i];
                sum += A_row[i+1] * B_shared[i+1];
                sum += A_row[i+2] * B_shared[i+2];
                sum += A_row[i+3] * B_shared[i+3];
            }
            // Handle remaining elements
            for (; i < elements; i++) {
                sum += A_row[i] * B_shared[i];
            }
        }
        __syncthreads();
    }
    
    if (row < M) {
        C[row] = sum;
    }
}

torch::Tensor matvec_hip(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    auto C = torch::zeros({M, 1}, A.options());
    
    // Use larger blocks for better occupancy
    const int block_size = 512;
    const int grid_size = (M + block_size - 1) / block_size;
    
    matvec_kernel<<<grid_size, block_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M,
        K
    );
    
    return C;
}
"""

matvec_module = load_inline(
    name="matvec",
    cpp_sources=matvec_cpp_source,
    functions=["matvec_hip"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matvec = matvec_module
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matvec.matvec_hip(A, B)