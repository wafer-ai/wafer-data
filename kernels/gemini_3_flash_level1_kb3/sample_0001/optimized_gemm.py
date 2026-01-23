
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

gemm_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 8

__global__ void __launch_bounds__(256) optimized_gemm_kernel_v4(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K, int N) {
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x % 16;
    int ty = threadIdx.x / 16;
    int tid = threadIdx.x;

    float rC[TM][TN] = {0.0f};

    for (int k = 0; k < K; k += BK) {
        // Load A into shared memory (128x16) - 2048 floats
        // 256 threads, each loads 8 floats = 2 float4
        for (int i = 0; i < 2; ++i) {
            int load_idx = tid * 2 + i;
            int row_a = load_idx / (BK / 4);
            int col_a = (load_idx % (BK / 4)) * 4;
            int global_row_a = by * BM + row_a;
            int global_col_a = k + col_a;
            if (global_row_a < M && global_col_a < K) {
                *((float4*)(&sA[row_a][col_a])) = *((float4*)(&A[global_row_a * K + global_col_a]));
            } else {
                *((float4*)(&sA[row_a][col_a])) = make_float4(0, 0, 0, 0);
            }
        }

        // Load B into shared memory (16x128) - 2048 floats
        // 256 threads, each loads 8 floats = 2 float4
        for (int i = 0; i < 2; ++i) {
            int load_idx = tid * 2 + i;
            int row_b = load_idx / (BN / 4);
            int col_b = (load_idx % (BN / 4)) * 4;
            int global_row_b = k + row_b;
            int global_col_b = bx * BN + col_b;
            if (global_row_b < K && global_col_b < N) {
                *((float4*)(&sB[row_b][col_b])) = *((float4*)(&B[global_row_b * N + global_col_b]));
            } else {
                *((float4*)(&sB[row_b][col_b])) = make_float4(0, 0, 0, 0);
            }
        }

        __syncthreads();

        #pragma unroll
        for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {
            float rA[TM];
            float rB[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) rA[i] = sA[ty * TM + i][dot_idx];
            #pragma unroll
            for (int j = 0; j < TN; ++j) rB[j] = sB[dot_idx][tx * TN + j];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    rC[i][j] += rA[i] * rB[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int global_row = by * BM + ty * TM + i;
            int global_col = bx * BN + tx * TN + j;
            if (global_row < M && global_col < N) {
                *((float4*)(&C[global_row * N + global_col])) = make_float4(rC[i][j], rC[i][j+1], rC[i][j+2], rC[i][j+3]);
            }
        }
    }
}

torch::Tensor optimized_gemm_hip(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    dim3 block_size(256);
    dim3 grid_size((N + BN - 1) / BN, (M + BM - 1) / BM);

    optimized_gemm_kernel_v4<<<grid_size, block_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );

    return C;
}
"""

gemm_module = load_inline(
    name="optimized_gemm_v4",
    cpp_sources=gemm_cpp_source,
    functions=["optimized_gemm_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gemm = gemm_module

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if A.is_cuda:
            return self.gemm.optimized_gemm_hip(A, B)
        else:
            return torch.matmul(A, B)
