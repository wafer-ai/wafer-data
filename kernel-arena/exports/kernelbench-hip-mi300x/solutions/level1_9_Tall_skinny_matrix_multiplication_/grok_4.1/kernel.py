import os
os.environ["CXX"] = "hipcc"
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

gemm_cpp = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
__global__ void gemm_kernel(const float *A, const float *B, float *C, int M, int K, int N) {
    constexpr int BM_THREADS = 32;
    constexpr int BN_THREADS = 32;
    constexpr int RM = 8;
    constexpr int BM = BM_THREADS * RM;
    constexpr int BN = BN_THREADS;
    __shared__ float shA[256][33];
    __shared__ float shB[32][33];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Load shA - multiple rows per thread along M
    for (int ll = 0; ll < RM; ++ll) {
        int row_sh = ty * RM + ll;
        if (row_sh < BM && tx < K) {
            int gi = by * BM + row_sh;
            if (gi < M) {
                shA[row_sh][tx] = A[gi * K + tx];
            }
        }
    }
    // Load shB - one per thread
    if (ty < K && tx < BN_THREADS && bx * BN + tx < N) {
        shB[ty][tx] = B[ty * N + bx * BN + tx];
    }
    __syncthreads();

    // Compute
    int base_sh_row = ty * RM;
    int base_i = by * BM + base_sh_row;
    int j = bx * BN + tx;
    for (int rm = 0; rm < RM; ++rm) {
        int ii = base_i + rm;
        if (ii < M) {
            float sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < K; ++k) {
                sum += shA[base_sh_row + rm][k] * shB[k][tx];
            }
            C[ii * N + j] = sum;
        }
    }
}
torch::Tensor gemm_hip(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    constexpr int BM_THREADS = 32;
    constexpr int BN_THREADS = 32;
    constexpr int RM = 8;
    constexpr int BM = BM_THREADS * RM;
    constexpr int BN = BN_THREADS;
    dim3 block(BN_THREADS, BM_THREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                 C.data_ptr<float>(), (int)M, (int)K, (int)N);
    return C;
}
"""

gemm = load_inline(
    name="gemm",
    cpp_sources=gemm_cpp,
    functions=["gemm_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gemm = gemm

    def forward(self, A, B):
        return self.gemm.gemm_hip(A, B)

M = 16384 * 2
N = 16 * 2

def get_inputs():
    A = torch.rand(M, N)
    B = torch.rand(N, M)
    return [A, B]

def get_init_inputs():
    return []
