import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

matvec_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define WARP_SIZE 64
#define BLOCK_SIZE 256

// Single-pass kernel with atomic adds for final accumulation
__global__ void matvec_atomic_kernel(const float* __restrict__ A, 
                                      const float* __restrict__ B, 
                                      float* __restrict__ C, 
                                      int M, int K, int blocks_per_row) {
    int row = blockIdx.x / blocks_per_row;
    int block_in_row = blockIdx.x % blocks_per_row;
    
    if (row >= M) return;
    
    // Calculate the portion of K this block handles
    int k_per_block = (K + blocks_per_row - 1) / blocks_per_row;
    int k_start = block_in_row * k_per_block;
    int k_end = min(k_start + k_per_block, K);
    
    const float* A_row = A + (size_t)row * K;
    
    float sum = 0.0f;
    
    // Vectorized access with float4
    int vec_k_start = (k_start + 3) / 4;  // Round up to alignment
    int vec_k_end = k_end / 4;            // Round down
    
    // Handle unaligned start
    for (int i = k_start + threadIdx.x; i < vec_k_start * 4 && i < k_end; i += blockDim.x) {
        sum += A_row[i] * B[i];
    }
    
    const float4* A_row_vec = reinterpret_cast<const float4*>(A_row);
    const float4* B_vec = reinterpret_cast<const float4*>(B);
    
    for (int i = vec_k_start + threadIdx.x; i < vec_k_end; i += blockDim.x) {
        float4 a_val = A_row_vec[i];
        float4 b_val = B_vec[i];
        sum += a_val.x * b_val.x + a_val.y * b_val.y + a_val.z * b_val.z + a_val.w * b_val.w;
    }
    
    // Handle remainder
    int remainder_start = vec_k_end * 4;
    for (int i = remainder_start + threadIdx.x; i < k_end; i += blockDim.x) {
        sum += A_row[i] * B[i];
    }
    
    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset);
    }
    
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    constexpr int num_warps = BLOCK_SIZE / WARP_SIZE;
    
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down(sum, offset);
        }
        
        if (lane == 0) {
            atomicAdd(&C[row], sum);
        }
    }
}

torch::Tensor matvec_hip_v5(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    
    auto C = torch::zeros({M, 1}, A.options());
    
    // Use more blocks for larger K
    int blocks_per_row = 32;  // Tune this
    
    dim3 block(BLOCK_SIZE);
    dim3 grid(M * blocks_per_row);
    
    matvec_atomic_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, blocks_per_row
    );
    
    return C;
}
"""

matvec_cpp_source = """
torch::Tensor matvec_hip_v5(torch::Tensor A, torch::Tensor B);
"""

matvec_module = load_inline(
    name="matvec_hip_v5",
    cpp_sources=matvec_cpp_source,
    cuda_sources=matvec_hip_source,
    functions=["matvec_hip_v5"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matvec = matvec_module
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matvec.matvec_hip_v5(A, B)
