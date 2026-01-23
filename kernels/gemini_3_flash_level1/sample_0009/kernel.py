
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

gemm_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define THREAD_M 4
#define THREAD_N 4

__global__ void gemm_kernel_final(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    __shared__ float sA[TILE_M][TILE_K];
    __shared__ float sB[TILE_K][TILE_N];

    int tx = threadIdx.x % 16;
    int ty = threadIdx.x / 16;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    float res[THREAD_M][THREAD_N];
    for (int i = 0; i < THREAD_M; i++) {
        for (int j = 0; j < THREAD_N; j++) {
            res[i][j] = 0.0f;
        }
    }

    for (int k_off = 0; k_off < K; k_off += TILE_K) {
        int a_idx = threadIdx.x * 4;
        int a_row = a_idx / TILE_K;
        int a_col = a_idx % TILE_K;
        int g_row = by * TILE_M + a_row;
        int g_col = k_off + a_col;
        
        if (g_row < M && g_col + 3 < K) {
            float4 val = reinterpret_cast<const float4*>(&A[g_row * K + g_col])[0];
            sA[a_row][a_col] = val.x;
            sA[a_row][a_col+1] = val.y;
            sA[a_row][a_col+2] = val.z;
            sA[a_row][a_col+3] = val.w;
        } else {
            for(int i=0; i<4; i++) {
                if (g_row < M && g_col + i < K)
                    sA[a_row][a_col+i] = A[g_row * K + g_col + i];
                else
                    sA[a_row][a_col+i] = 0.0f;
            }
        }

        int b_idx = threadIdx.x * 4;
        int b_row = b_idx / TILE_N;
        int b_col = b_idx % TILE_N;
        int gb_row = k_off + b_row;
        int gb_col = bx * TILE_N + b_col;

        if (gb_row < K && gb_col + 3 < N) {
            float4 val = reinterpret_cast<const float4*>(&B[gb_row * N + gb_col])[0];
            sB[b_row][b_col] = val.x;
            sB[b_row][b_col+1] = val.y;
            sB[b_row][b_col+2] = val.z;
            sB[b_row][b_col+3] = val.w;
        } else {
            for(int i=0; i<4; i++) {
                if (gb_row < K && gb_col + i < N)
                    sB[b_row][b_col+i] = B[gb_row * N + gb_col + i];
                else
                    sB[b_row][b_col+i] = 0.0f;
            }
        }

        __syncthreads();

        for (int k = 0; k < TILE_K; k++) {
            float a_cache[THREAD_M];
            for (int i = 0; i < THREAD_M; i++) a_cache[i] = sA[ty * THREAD_M + i][k];
            float b_cache[THREAD_N];
            for (int j = 0; j < THREAD_N; j++) b_cache[j] = sB[k][tx * THREAD_N + j];
            for (int i = 0; i < THREAD_M; i++) {
                for (int j = 0; j < THREAD_N; j++) {
                    res[i][j] += a_cache[i] * b_cache[j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < THREAD_M; i++) {
        for (int j = 0; j < THREAD_N; j++) {
            int row = by * TILE_M + ty * THREAD_M + i;
            int col = bx * TILE_N + tx * THREAD_N + j;
            if (row < M && col < N) {
                C[row * N + col] = res[i][j];
            }
        }
    }
}

torch::Tensor gemm_hip(torch::Tensor A, torch::Tensor B) {
    auto A_cont = A.reshape({-1, A.size(-1)}).contiguous();
    auto B_cont = B.contiguous();
    int M = A_cont.size(0);
    int K = A_cont.size(1);
    int N = B_cont.size(1);
    auto C = torch::empty({M, N}, A.options());
    dim3 block_size(256);
    dim3 num_blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    gemm_kernel_final<<<num_blocks, block_size>>>(A_cont.data_ptr<float>(), B_cont.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C.view({A.size(0), A.size(1), A.size(2), N});
}
"""

gemm_module = load_inline(
    name="gemm_module_final",
    cpp_sources=gemm_cpp_source,
    functions=["gemm_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gemm_module = gemm_module

    def forward(self, A, B):
        return self.gemm_module.gemm_hip(A, B)
