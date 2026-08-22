import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>

#define ROWS_PER_BLOCK 8
#define BLOCK_SIZE 256

__global__ void __launch_bounds__(BLOCK_SIZE) gemv_splitk_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K) {
    int block_row_start = blockIdx.x * ROWS_PER_BLOCK;
    int split_idx = blockIdx.y;
    int num_splits = gridDim.y;
    
    // Calculate K range for this split
    // Assume K is divisible by num_splits for simplicity in this optimization
    int k_chunk = K / num_splits; 
    int k_start = split_idx * k_chunk;
    int k_end = k_start + k_chunk;
    
    // Ensure last split covers everything if not perfectly divisible
    if (split_idx == num_splits - 1) {
        k_end = K;
    }
    
    // Output accumulators
    float sum[ROWS_PER_BLOCK];
    #pragma unroll
    for (int i = 0; i < ROWS_PER_BLOCK; ++i) {
        sum[i] = 0.0f;
    }

    // Shared memory for reduction
    __shared__ float sdata[ROWS_PER_BLOCK][BLOCK_SIZE];

    int tid = threadIdx.x;
    
    const float4* B_vec = reinterpret_cast<const float4*>(B);
    const float4* A_vec = reinterpret_cast<const float4*>(A);

    int k_vec_start = k_start / 4;
    int k_vec_end = k_end / 4;
    
    // Precompute row offsets (in float4 elements)
    size_t row_offsets[ROWS_PER_BLOCK];
    #pragma unroll
    for (int i = 0; i < ROWS_PER_BLOCK; ++i) {
        int current_row = block_row_start + i;
        row_offsets[i] = (size_t)current_row * (K / 4);
    }

    // Main loop
    #pragma unroll 4
    for (int k = k_vec_start + tid; k < k_vec_end; k += BLOCK_SIZE) {
        float4 b_val = B_vec[k];
        
        #pragma unroll
        for (int i = 0; i < ROWS_PER_BLOCK; ++i) {
            // Check M bounds. 
            int current_row = block_row_start + i;
            if (current_row < M) {
                float4 a_val = A_vec[row_offsets[i] + k];
                
                sum[i] += a_val.x * b_val.x;
                sum[i] += a_val.y * b_val.y;
                sum[i] += a_val.z * b_val.z;
                sum[i] += a_val.w * b_val.w;
            }
        }
    }

    // Store sums to shared memory
    #pragma unroll
    for (int i = 0; i < ROWS_PER_BLOCK; ++i) {
        sdata[i][tid] = sum[i];
    }
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            #pragma unroll
            for (int i = 0; i < ROWS_PER_BLOCK; ++i) {
                sdata[i][tid] += sdata[i][tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < ROWS_PER_BLOCK; ++i) {
            int current_row = block_row_start + i;
            if (current_row < M) {
                atomicAdd(&C[current_row], sdata[i][0]);
            }
        }
    }
}

torch::Tensor gemv_hip(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    
    auto C = torch::zeros({M, 1}, A.options());
    
    // Tuning parameters
    int split_k = 16;
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, split_k);
    
    gemv_splitk_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K);
    
    return C;
}
"""

gemv_module = load_inline(
    name="gemv_module",
    cpp_sources=cpp_source,
    functions=["gemv_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gemv_hip = gemv_module.gemv_hip
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.gemv_hip(A, B)

M = 256 * 8 # 2048
K = 131072 * 8 # 1048576

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, 1)
    return [A, B]

def get_init_inputs():
    return []
