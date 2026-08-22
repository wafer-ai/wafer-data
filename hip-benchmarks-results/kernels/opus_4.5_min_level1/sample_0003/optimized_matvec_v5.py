import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

matvec_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define WARP_SIZE 64
#define BLOCK_SIZE 1024
#define TILE_SIZE 4096  // Tile size for B vector caching

// Process multiple rows per block using shared memory caching of B
__global__ void matvec_tiled_kernel(const float* __restrict__ A, 
                                     const float* __restrict__ B, 
                                     float* __restrict__ C,
                                     int M, int K) {
    int row = blockIdx.x;
    if (row >= M) return;
    
    const float* A_row = A + row * K;
    
    __shared__ float B_shared[TILE_SIZE];
    
    float sum = 0.0f;
    
    // Process K in tiles
    for (int tile_start = 0; tile_start < K; tile_start += TILE_SIZE) {
        int tile_end = min(tile_start + TILE_SIZE, K);
        int tile_len = tile_end - tile_start;
        
        // Cooperatively load B tile into shared memory
        for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
            B_shared[i] = B[tile_start + i];
        }
        __syncthreads();
        
        // Each thread processes part of the tile using float4
        const float4* A_tile_f4 = reinterpret_cast<const float4*>(A_row + tile_start);
        const float4* B_tile_f4 = reinterpret_cast<const float4*>(B_shared);
        
        int num_float4 = tile_len / 4;
        for (int i = threadIdx.x; i < num_float4; i += blockDim.x) {
            float4 a = A_tile_f4[i];
            float4 b = B_tile_f4[i];
            sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
        
        // Handle remaining elements in tile
        int remainder_start = num_float4 * 4;
        for (int i = remainder_start + threadIdx.x; i < tile_len; i += blockDim.x) {
            sum += A_row[tile_start + i] * B_shared[i];
        }
        
        __syncthreads();
    }
    
    // Warp-level reduction
    __shared__ float shared_data[BLOCK_SIZE / WARP_SIZE];
    
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset, WARP_SIZE);
    }
    
    if (lane == 0) {
        shared_data[warp_id] = sum;
    }
    __syncthreads();
    
    int num_warps = BLOCK_SIZE / WARP_SIZE;
    if (warp_id == 0) {
        sum = (lane < num_warps) ? shared_data[lane] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down(sum, offset, WARP_SIZE);
        }
        if (lane == 0) {
            C[row] = sum;
        }
    }
}

torch::Tensor matvec_hip(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    
    auto C = torch::zeros({M, 1}, A.options());
    
    dim3 block(BLOCK_SIZE);
    dim3 grid(M);
    
    matvec_tiled_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K
    );
    
    return C;
}
"""

matvec_cpp_source = """
torch::Tensor matvec_hip(torch::Tensor A, torch::Tensor B);
"""

matvec_module = load_inline(
    name="matvec_hip",
    cpp_sources=matvec_cpp_source,
    cuda_sources=matvec_hip_source,
    functions=["matvec_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matvec = matvec_module

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matvec.matvec_hip(A, B)
