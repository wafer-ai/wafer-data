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
#define BLOCKS_PER_ROW 32

// Use atomicAdd for final reduction instead of two-kernel approach
__global__ void matvec_atomic_kernel(const float* __restrict__ A, 
                                      const float* __restrict__ B, 
                                      float* __restrict__ C,
                                      int M, int K) {
    int blocks_per_row = BLOCKS_PER_ROW;
    int row = blockIdx.x / blocks_per_row;
    int block_in_row = blockIdx.x % blocks_per_row;
    
    if (row >= M) return;
    
    const float* A_row = A + row * K;
    
    // Calculate range this block handles
    int chunk_size = (K + blocks_per_row - 1) / blocks_per_row;
    int start = block_in_row * chunk_size;
    int end = min(start + chunk_size, K);
    
    if (start >= K) return;
    
    float sum = 0.0f;
    
    // Vectorized processing with float4
    const float4* A_row_f4 = reinterpret_cast<const float4*>(A_row);
    const float4* B_f4 = reinterpret_cast<const float4*>(B);
    
    int start_f4 = (start + 3) / 4;  // Round up
    int end_f4 = end / 4;            // Round down
    
    // Handle pre-aligned elements
    for (int i = start + threadIdx.x; i < start_f4 * 4 && i < end; i += blockDim.x) {
        sum += A_row[i] * B[i];
    }
    
    // Main vectorized loop
    for (int i = start_f4 + threadIdx.x; i < end_f4; i += blockDim.x) {
        float4 a = A_row_f4[i];
        float4 b = B_f4[i];
        sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    
    // Handle post-aligned elements
    for (int i = end_f4 * 4 + threadIdx.x; i < end; i += blockDim.x) {
        sum += A_row[i] * B[i];
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
            atomicAdd(&C[row], sum);
        }
    }
}

torch::Tensor matvec_hip(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    
    auto C = torch::zeros({M, 1}, A.options());
    
    dim3 block(BLOCK_SIZE);
    dim3 grid(M * BLOCKS_PER_ROW);
    
    matvec_atomic_kernel<<<grid, block>>>(
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
