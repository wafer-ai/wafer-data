
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

mv_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define THREADS_PER_BLOCK 256
#define ROWS_PER_BLOCK 4
#define WARP_SIZE 64

__global__ void mv_kernel_multi_row(const float4* __restrict__ A, const float4* __restrict__ B, float* __restrict__ C, int M, int K_vec) {
    int start_row = blockIdx.x * ROWS_PER_BLOCK;
    int tid = threadIdx.x;

    float sum[ROWS_PER_BLOCK];
    for (int i = 0; i < ROWS_PER_BLOCK; i++) sum[i] = 0.0f;

    for (int k = tid; k < K_vec; k += THREADS_PER_BLOCK) {
        float4 b = B[k];
        for (int i = 0; i < ROWS_PER_BLOCK; i++) {
            if (start_row + i < M) {
                float4 a = A[(start_row + i) * K_vec + k];
                sum[i] += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }
        }
    }

    __shared__ float shared_sum[ROWS_PER_BLOCK][THREADS_PER_BLOCK / WARP_SIZE];
    int warpId = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    for (int i = 0; i < ROWS_PER_BLOCK; i++) {
        float val = sum[i];
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
            val += __shfl_down(val, offset);
        if (lane == 0) shared_sum[i][warpId] = val;
    }
    __syncthreads();

    if (tid < WARP_SIZE) {
        for (int i = 0; i < ROWS_PER_BLOCK; i++) {
            if (start_row + i < M) {
                float val = (tid < (THREADS_PER_BLOCK / WARP_SIZE)) ? shared_sum[i][tid] : 0.0f;
                for (int offset = (THREADS_PER_BLOCK / WARP_SIZE) / 2; offset > 0; offset /= 2)
                    val += __shfl_down(val, offset);
                if (tid == 0) C[start_row + i] = val;
            }
        }
    }
}

torch::Tensor mv_hip(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto C = torch::empty({M, 1}, A.options());

    int K_vec = K / 4;
    int num_blocks = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

    mv_kernel_multi_row<<<num_blocks, THREADS_PER_BLOCK>>>(
        (const float4*)A.data_ptr<float>(),
        (const float4*)B.data_ptr<float>(),
        C.data_ptr<float>(),
        M,
        K_vec
    );

    return C;
}
"""

mv_module = load_inline(
    name="mv_module",
    cpp_sources=mv_cpp_source,
    functions=["mv_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mv_module = mv_module

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.mv_module.mv_hip(A, B)

