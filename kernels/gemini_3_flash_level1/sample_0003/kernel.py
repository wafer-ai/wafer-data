
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

gemv_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 32; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset, 64);
    }
    return val;
}

__global__ void gemv_kernel_v5(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K) {
    int row_group = blockIdx.x; // Each block group handles 8 rows
    int row_start = row_group * 8;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    float sum[8] = {0.0f};

    const float4* B4 = reinterpret_cast<const float4*>(B);
    const float4* A_4[8];
    for(int j=0; j<8; ++j) {
        A_4[j] = reinterpret_cast<const float4*>(A + (row_start + j) * K);
    }

    int K4 = K / 4;
    for (int i = tid; i < K4; i += blockSize) {
        float4 b_val = B4[i];
        #pragma unroll
        for(int j=0; j<8; ++j) {
            float4 a_val = A_4[j][i];
            sum[j] += a_val.x * b_val.x + a_val.y * b_val.y + a_val.z * b_val.z + a_val.w * b_val.w;
        }
    }

    // Reduction using warp shuffles
    #pragma unroll
    for(int j=0; j<8; ++j) {
        sum[j] = warpReduceSum(sum[j]);
    }

    static __shared__ float shared_sum[8][4]; // 256 threads / 64 = 4 warps
    int lane = tid % 64;
    int warpId = tid / 64;

    if (lane == 0) {
        #pragma unroll
        for(int j=0; j<8; ++j) {
            shared_sum[j][warpId] = sum[j];
        }
    }
    __syncthreads();

    if (tid < 64) {
        #pragma unroll
        for(int j=0; j<8; ++j) {
            float val = (tid < 4) ? shared_sum[j][tid] : 0.0f;
            val = warpReduceSum(val);
            if (tid == 0) {
                C[row_start + j] = val;
            }
        }
    }
}

torch::Tensor gemv_hip(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    auto C = torch::empty({M, 1}, A.options());

    int blockSize = 256;
    int rowsPerBlock = 8;
    int numBlocks = M / rowsPerBlock;
    
    gemv_kernel_v5<<<numBlocks, blockSize>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        M, K
    );

    return C;
}
"""

gemv_module = load_inline(
    name="gemv_module_v5",
    cpp_sources=gemv_kernel_source,
    functions=["gemv_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gemv_hip = gemv_module.gemv_hip

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.gemv_hip(A, B)
