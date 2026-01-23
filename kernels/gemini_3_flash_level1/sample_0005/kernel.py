
import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

gemm_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 8

__global__ void __launch_bounds__(256)
optimized_gemm_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    __shared__ float sA[BK][BM]; 
    __shared__ float sB[BK][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int tx = tid % 16;
    int ty = tid / 16;

    float rC[TM][TN] = {0.0f};

    for (int k_off = 0; k_off < N; k_off += BK) {
        // Corrected shared memory loading: each of 256 threads loads 8 elements
        for (int i = 0; i < 8; ++i) {
            int load_tid = tid + i * 256;
            
            // For sA (BKxBM = 16x128 = 2048 elements)
            int row_a_sh = load_tid % BM;
            int col_a_sh = load_tid / BM;
            sA[col_a_sh][row_a_sh] = A[(by * BM + row_a_sh) * N + (k_off + col_a_sh)];
            
            // For sB (BKxBN = 16x128 = 2048 elements)
            int row_b_sh = load_tid / BN;
            int col_b_sh = load_tid % BN;
            sB[row_b_sh][col_b_sh] = B[(k_off + row_b_sh) * N + (bx * BN + col_b_sh)];
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float rA[TM];
            float rB[TN];
            #pragma unroll
            for (int m = 0; m < TM; ++m) rA[m] = sA[k][ty * TM + m];
            #pragma unroll
            for (int n = 0; n < TN; ++n) rB[n] = sB[k][tx * TN + n];
            
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    rC[m][n] += rA[m] * rB[n];
                }
            }
        }
        __syncthreads();
    }

    for (int m = 0; m < TM; ++m) {
        for (int n = 0; n < TN; ++n) {
            C[(by * BM + ty * TM + m) * N + (bx * BN + tx * TN + n)] = rC[m][n];
        }
    }
}

torch::Tensor optimized_gemm_hip(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::empty({N, N}, A.options());
    dim3 threadsPerBlock(256);
    dim3 numBlocks(N / BN, N / BM);
    optimized_gemm_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        N
    );
    return C;
}
"""

gemm_module = load_inline(
    name="optimized_gemm_final_v2",
    cpp_sources=gemm_cpp_source,
    functions=["optimized_gemm_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gemm_module = gemm_module

    def forward(self, A, B):
        return self.gemm_module.optimized_gemm_hip(A, B)

N = 4096
def get_inputs():
    A = torch.rand(N, N)
    A = (A + A.T) / 2
    B = torch.rand(N, N)
    B = (B + B.T) / 2
    return [A.cuda(), B.cuda()]

def get_init_inputs():
    return []
