
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

mvm_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void mvm_kernel_v5(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K) {
    int part = blockIdx.x;
    int row = blockIdx.y;
    int num_parts = gridDim.x;

    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    
    long long elements_per_part = (K + num_parts - 1) / num_parts;
    long long start_idx = (long long)part * elements_per_part;
    long long end_idx = min(start_idx + elements_per_part, (long long)K);
    
    float sum = 0.0f;
    const float* row_A = A + (long long)row * K;

    // Vectorized part with unrolling
    long long i = start_idx + tid * 8;
    for (; i + 7 < end_idx; i += blockSize * 8) {
        float4 a1 = *reinterpret_cast<const float4*>(row_A + i);
        float4 b1 = *reinterpret_cast<const float4*>(B + i);
        float4 a2 = *reinterpret_cast<const float4*>(row_A + i + 4);
        float4 b2 = *reinterpret_cast<const float4*>(B + i + 4);
        sum += a1.x * b1.x + a1.y * b1.y + a1.z * b1.z + a1.w * b1.w;
        sum += a2.x * b2.x + a2.y * b2.y + a2.z * b2.z + a2.w * b2.w;
    }
    // Handle remaining elements
    for (long long j = start_idx + ((end_idx - start_idx) / (blockSize * 8)) * (blockSize * 8) + tid; j < end_idx; j += blockSize) {
        sum += row_A[j] * B[j];
    }

    // Wavefront reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    __shared__ float shared_sum[32]; 
    int lane_id = tid % warpSize;
    int wave_id = tid / warpSize;

    if (lane_id == 0) {
        shared_sum[wave_id] = sum;
    }
    __syncthreads();

    if (tid < (blockSize / warpSize)) {
        float s = shared_sum[tid];
        for (int offset = (blockSize / warpSize) / 2; offset > 0; offset >>= 1) {
            s += __shfl_down(s, offset);
        }
        if (tid == 0) {
            atomicAdd(&C[row], s);
        }
    }
}

torch::Tensor mvm_hip(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    auto C = torch::zeros({M, 1}, A.options());

    int block_size = 256;
    int num_blocks_per_row = 32; 
    dim3 grid(num_blocks_per_row, M);

    mvm_kernel_v5<<<grid, block_size>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K);

    return C;
}
"""

mvm_lib = load_inline(
    name="mvm_lib_v5",
    cpp_sources=mvm_cpp_source,
    functions=["mvm_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.mvm_lib = mvm_lib

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.mvm_lib.mvm_hip(A, B)
