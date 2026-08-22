import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>

#define BM 64
#define BN 64
#define BK 32
#define TM 4
#define TN 4
#define BK_PAD 40

__global__ void sgemm_64x64_bk32(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    __shared__ float As[BM][BK_PAD];
    __shared__ float Bs[BK][BN];

    float c_reg[TM][TN] = {0.0f};
    float a_reg[TM];
    float b_reg[TN];

    const float* A_ptr = A + by * BM * K;
    const float* B_ptr = B + bx * BN;

    for (int k = 0; k < K; k += BK) {
        // Load A: 2 float4s per thread
        #pragma unroll
        for (int l = 0; l < 2; ++l) {
            int f4_idx = tid * 2 + l;
            int row = f4_idx / (BK/4); 
            int col = (f4_idx % (BK/4)) * 4;
            
            float4 vec = *reinterpret_cast<const float4*>(&A_ptr[row * K + k + col]);
            *reinterpret_cast<float4*>(&As[row][col]) = vec;
        }

        // Load B: 2 float4s per thread
        #pragma unroll
        for (int l = 0; l < 2; ++l) {
            int f4_idx = tid * 2 + l;
            int row = f4_idx / (BN/4);
            int col = (f4_idx % (BN/4)) * 4;
            
            float4 vec = *reinterpret_cast<const float4*>(&B_ptr[(k + row) * N + col]);
            *reinterpret_cast<float4*>(&Bs[row][col]) = vec;
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < BK; ++i) {
            #pragma unroll
            for (int r = 0; r < TM; ++r) {
                a_reg[r] = As[ty * TM + r][i];
            }
            
            *reinterpret_cast<float4*>(&b_reg[0]) = *reinterpret_cast<float4*>(&Bs[i][tx * TN]);

            #pragma unroll
            for (int r = 0; r < TM; ++r) {
                #pragma unroll
                for (int c = 0; c < TN; ++c) {
                    c_reg[r][c] += a_reg[r] * b_reg[c];
                }
            }
        }

        __syncthreads();
    }

    int c_global_row = by * BM + ty * TM;
    int c_global_col = bx * BN + tx * TN;

    #pragma unroll
    for (int r = 0; r < TM; ++r) {
        float4 vec;
        vec.x = c_reg[r][0]; vec.y = c_reg[r][1]; vec.z = c_reg[r][2]; vec.w = c_reg[r][3];
        *reinterpret_cast<float4*>(&C[(c_global_row + r) * N + c_global_col]) = vec;
    }
}

torch::Tensor matmul_custom(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 dimBlock(16, 16);
    dim3 dimGrid(N / 64, M / 64);

    sgemm_64x64_bk32<<<dimGrid, dimBlock>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    
    return C;
}
"""

module = load_inline(
    name="custom_matmul_v5",
    cpp_sources=cpp_source,
    functions=["matmul_custom"],
    extra_cflags=['-O3', '-ffast-math'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.module = module
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.module.matmul_custom(A, B)

M = 2048
K = 8192
N = 4096

def get_inputs():
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    return [A, B]

def get_init_inputs():
    return []
