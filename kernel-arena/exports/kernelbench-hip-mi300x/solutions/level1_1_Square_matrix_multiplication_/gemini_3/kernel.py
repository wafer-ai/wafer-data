
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ['CXX'] = 'hipcc'

cpp_source = """
#include <hip/hip_runtime.h>

#define BM 64
#define BN 64
#define BK 32
#define TM 4
#define TN 8

__global__ void matmul_opt_4x8_bk32(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * 8 + tx; // blockDim.x = 8

    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BK][BN + 1];

    float thread_C[TM][TN] = {0.0f};
    float a_reg[TM];
    float b_reg[TN];

    // Load indices
    // 128 threads.
    // As: 64x32 = 2048 floats = 512 float4. 4 per thread.
    int a_vec_base = tid * 4;
    // Bs: 32x64 = 2048 floats = 512 float4. 4 per thread.
    int b_vec_base = tid * 4;

    const float* A_ptr = A + by * BM * N;
    // B ptr dynamic

    for (int k = 0; k < N; k += BK) {
        // Unrolled loads
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
             int vec = a_vec_base + i;
             int r = vec / 8; // 64 rows, 8 vectors per row (32 cols / 4)
             int c = (vec % 8) * 4;
             
             float4 val = *reinterpret_cast<const float4*>(&A_ptr[r * N + k + c]);
             As[r][c + 0] = val.x;
             As[r][c + 1] = val.y;
             As[r][c + 2] = val.z;
             As[r][c + 3] = val.w;
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
             int vec = b_vec_base + i;
             int r = vec / 16; // 32 rows, 16 vectors per row (64 cols / 4)
             int c = (vec % 16) * 4;
             
             float4 val = *reinterpret_cast<const float4*>(&B[(k + r) * N + bx * BN + c]);
             Bs[r][c + 0] = val.x;
             Bs[r][c + 1] = val.y;
             Bs[r][c + 2] = val.z;
             Bs[r][c + 3] = val.w;
        }

        __syncthreads();

        #pragma unroll
        for (int l = 0; l < BK; ++l) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_reg[i] = As[ty * TM + i][l];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_reg[j] = Bs[l][tx * TN + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    thread_C[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    int c_row = by * BM + ty * TM;
    int c_col = bx * BN + tx * TN;

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        // Store 8 floats (2 float4)
        float4 c0, c1;
        c0.x = thread_C[i][0]; c0.y = thread_C[i][1]; c0.z = thread_C[i][2]; c0.w = thread_C[i][3];
        c1.x = thread_C[i][4]; c1.y = thread_C[i][5]; c1.z = thread_C[i][6]; c1.w = thread_C[i][7];
        
        *reinterpret_cast<float4*>(&C[(c_row + i) * N + c_col]) = c0;
        *reinterpret_cast<float4*>(&C[(c_row + i) * N + c_col + 4]) = c1;
    }
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::empty({N, N}, A.options());

    dim3 block(8, 16);
    dim3 grid(N / BN, N / BM);

    matmul_opt_4x8_bk32<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}
"""

matmul_module = load_inline(
    name="matmul_custom",
    cpp_sources=cpp_source,
    functions=["matmul_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_op = matmul_module

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_op.matmul_hip(A, B)
