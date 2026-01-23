
import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

matmul_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

const int BM = 128;
const int BN = 128;
const int BK = 32;
const int TM = 8;
const int TN = 8;

__global__ void __launch_bounds__(256) matmul_kernel_optimized_v4(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    __shared__ float sA[BK][BM];
    __shared__ float sB[BK][BN];

    float rC[TM][TN] = {0.0f};
    float rA[TM];
    float rB[TN];

    int tid = threadIdx.x;
    int tx = tid % (BN / TN);
    int ty = tid / (BN / TN);

    for (int k = 0; k < N; k += BK) {
        // Load A into sA[BK][BM]
        // BM * BK = 128 * 32 = 4096 elements. 4096 / 4 = 1024 float4s.
        // Each of 256 threads loads 4 float4s.
        for (int i = 0; i < 4; ++i) {
            int load_idx = tid + i * 256;
            int load_row = load_idx / (BK / 4); // 0 to 127
            int load_col = (load_idx % (BK / 4)) * 4; // 0, 4, 8, ..., 28
            
            int g_row = blockIdx.y * BM + load_row;
            int g_col = k + load_col;

            if (g_row < N && g_col < N) {
                float4 tmp = reinterpret_cast<const float4*>(&A[g_row * N + g_col])[0];
                sA[load_col + 0][load_row] = tmp.x;
                sA[load_col + 1][load_row] = tmp.y;
                sA[load_col + 2][load_row] = tmp.z;
                sA[load_col + 3][load_row] = tmp.w;
            } else {
                sA[load_col + 0][load_row] = 0.0f;
                sA[load_col + 1][load_row] = 0.0f;
                sA[load_col + 2][load_row] = 0.0f;
                sA[load_col + 3][load_row] = 0.0f;
            }
        }

        // Load B into sB[BK][BN]
        // BK * BN = 32 * 128 = 4096 elements. 4096 / 4 = 1024 float4s.
        // Each of 256 threads loads 4 float4s.
        for (int i = 0; i < 4; ++i) {
            int load_idx = tid + i * 256;
            int load_row = load_idx / (BN / 4); // 0 to 31
            int load_col = (load_idx % (BN / 4)) * 4; // 0 to 124
            
            int g_row = k + load_row;
            int g_col = blockIdx.x * BN + load_col;

            if (g_row < N && g_col < N) {
                reinterpret_cast<float4*>(&sB[load_row][load_col])[0] = reinterpret_cast<const float4*>(&B[g_row * N + g_col])[0];
            } else {
                reinterpret_cast<float4*>(&sB[load_row][load_col])[0] = make_float4(0, 0, 0, 0);
            }
        }

        __syncthreads();

        for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {
            for (int i = 0; i < TM; ++i) {
                rA[i] = sA[dot_idx][ty * TM + i];
            }
            for (int i = 0; i < TN; ++i) {
                rB[i] = sB[dot_idx][tx * TN + i];
            }
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    rC[i][j] += rA[i] * rB[j];
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; j += 4) {
            int g_row = blockIdx.y * BM + ty * TM + i;
            int g_col = blockIdx.x * BN + tx * TN + j;
            if (g_row < N && g_col < N) {
                float4 tmp;
                tmp.x = rC[i][j + 0];
                tmp.y = rC[i][j + 1];
                tmp.z = rC[i][j + 2];
                tmp.w = rC[i][j + 3];
                reinterpret_cast<float4*>(&C[g_row * N + g_col])[0] = tmp;
            }
        }
    }
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::empty({N, N}, A.options());

    dim3 block_size(256);
    dim3 grid_size((N + BN - 1) / BN, (N + BM - 1) / BM);

    matmul_kernel_optimized_v4<<<grid_size, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}
"""

matmul_cuda = load_inline(
    name="matmul_hip_opt_4",
    cpp_sources=matmul_cpp_source,
    functions=["matmul_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_hip = matmul_cuda.matmul_hip

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if not A.is_cuda:
            A = A.cuda()
        if not B.is_cuda:
            B = B.cuda()
        return self.matmul_hip(A, B)

N = 2048 * 2

def get_inputs():
    A = torch.rand(N, N).cuda()
    B = torch.rand(N, N).cuda()
    return [A, B]

def get_init_inputs():
    return []
