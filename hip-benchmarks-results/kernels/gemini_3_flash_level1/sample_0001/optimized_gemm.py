
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

optimized_gemm_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void __launch_bounds__(256) optimized_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 16;
    const int TM = 8;
    const int TN = 8;

    const int thread_y = threadIdx.y;
    const int thread_x = threadIdx.x;
    const int block_y = blockIdx.y;
    const int block_x = blockIdx.x;

    // Use padding to avoid shared memory bank conflicts
    __shared__ float sA[BM][BK + 1];
    __shared__ float sB[BK][BN + 1];

    float rC[TM][TN];
    for(int i=0; i<TM; ++i) for(int j=0; j<TN; ++j) rC[i][j] = 0.0f;
    
    float rA[TM];
    float rB[TN];

    const int tid = thread_y * 16 + thread_x;
    const int tid_a_row = tid / 4; 
    const int tid_a_col = (tid % 4) * 4; 
    const int tid_b_row = tid / 32; 
    const int tid_b_col = (tid % 32) * 4; 

    for (int k = 0; k < K; k += BK) {
        // Load sA (vectorized) - 128x16 elements = 2048. 256 threads. Each 8 elements.
        // Wait, 256 * 4 = 1024. Need to load twice.
        for (int i = 0; i < 2; ++i) {
            int row = (tid / (BK / 4)) + i * (256 / (BK / 4));
            int col = (tid % (BK / 4)) * 4;
            if (row < BM) {
                float4 val = *(const float4*)&A[(block_y * BM + row) * K + (k + col)];
                sA[row][col] = val.x;
                sA[row][col+1] = val.y;
                sA[row][col+2] = val.z;
                sA[row][col+3] = val.w;
            }
        }

        // Load sB (vectorized) - 16x128 elements = 2048. 256 threads. Each 8 elements.
        for (int i = 0; i < 2; ++i) {
            int row = (tid / (BN / 4)) + i * (256 / (BN / 4));
            int col = (tid % (BN / 4)) * 4;
            if (row < BK) {
                float4 val = *(const float4*)&B[(k + row) * N + (block_x * BN + col)];
                sB[row][col] = val.x;
                sB[row][col+1] = val.y;
                sB[row][col+2] = val.z;
                sB[row][col+3] = val.w;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int ki = 0; ki < BK; ++ki) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) rA[i] = sA[thread_y * TM + i][ki];
            #pragma unroll
            for (int j = 0; j < TN; ++j) rB[j] = sB[ki][thread_x * TN + j];

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

    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            C[(block_y * BM + thread_y * TM + i) * N + (block_x * BN + thread_x * TN + j)] = rC[i][j];
        }
    }
}

torch::Tensor optimized_gemm_hip(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    
    dim3 block(16, 16);
    dim3 grid(N / 128, M / 128);

    optimized_gemm_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    return C;
}
"""

gemm_module = load_inline(
    name="optimized_gemm",
    cpp_sources=optimized_gemm_cpp_source,
    functions=["optimized_gemm_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gemm = gemm_module

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.gemm.optimized_gemm_hip(A, B)

def get_inputs():
    M, K, N = 2048, 8192, 4096
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []
