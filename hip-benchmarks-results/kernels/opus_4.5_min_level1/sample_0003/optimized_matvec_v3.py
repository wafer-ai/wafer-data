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
#define BLOCKS_PER_ROW 16  // Use multiple blocks per row for better parallelism

// First kernel: Partial reductions (multiple blocks per row)
__global__ void matvec_partial_kernel(const float* __restrict__ A, 
                                       const float* __restrict__ B, 
                                       float* __restrict__ partial_sums,
                                       int M, int K, int blocks_per_row) {
    int row = blockIdx.x / blocks_per_row;
    int block_in_row = blockIdx.x % blocks_per_row;
    
    if (row >= M) return;
    
    const float* A_row = A + row * K;
    
    // Calculate range this block will handle
    int elements_per_block = (K + blocks_per_row - 1) / blocks_per_row;
    int start = block_in_row * elements_per_block;
    int end = min(start + elements_per_block, K);
    
    float sum = 0.0f;
    
    // Process with float4 vectorization
    int aligned_start = ((start + 3) / 4) * 4;  // Round up to multiple of 4
    int aligned_end = (end / 4) * 4;            // Round down to multiple of 4
    
    // Handle unaligned start
    for (int i = start + threadIdx.x; i < aligned_start && i < end; i += blockDim.x) {
        sum += A_row[i] * B[i];
    }
    
    // Vectorized main loop
    const float4* A_row_f4 = reinterpret_cast<const float4*>(A_row);
    const float4* B_f4 = reinterpret_cast<const float4*>(B);
    
    for (int i = aligned_start/4 + threadIdx.x; i < aligned_end/4; i += blockDim.x) {
        float4 a_val = A_row_f4[i];
        float4 b_val = B_f4[i];
        sum += a_val.x * b_val.x + a_val.y * b_val.y + a_val.z * b_val.z + a_val.w * b_val.w;
    }
    
    // Handle unaligned end
    for (int i = aligned_end + threadIdx.x; i < end; i += blockDim.x) {
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
            partial_sums[row * blocks_per_row + block_in_row] = sum;
        }
    }
}

// Second kernel: Final reduction across blocks for each row
__global__ void matvec_final_kernel(const float* __restrict__ partial_sums,
                                     float* __restrict__ C,
                                     int M, int blocks_per_row) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    
    float sum = 0.0f;
    for (int i = 0; i < blocks_per_row; i++) {
        sum += partial_sums[row * blocks_per_row + i];
    }
    C[row] = sum;
}

torch::Tensor matvec_hip(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    
    auto options = A.options();
    auto C = torch::zeros({M, 1}, options);
    auto partial_sums = torch::zeros({M * BLOCKS_PER_ROW}, options);
    
    // First kernel: partial reductions
    dim3 block1(BLOCK_SIZE);
    dim3 grid1(M * BLOCKS_PER_ROW);
    
    matvec_partial_kernel<<<grid1, block1>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        M, K, BLOCKS_PER_ROW
    );
    
    // Second kernel: final reduction
    dim3 block2(256);
    dim3 grid2((M + 255) / 256);
    
    matvec_final_kernel<<<grid2, block2>>>(
        partial_sums.data_ptr<float>(),
        C.data_ptr<float>(),
        M, BLOCKS_PER_ROW
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
