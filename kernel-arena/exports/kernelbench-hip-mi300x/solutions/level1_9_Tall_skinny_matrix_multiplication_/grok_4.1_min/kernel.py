import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

matmul_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_F32(x) TORCH_CHECK(x.scalar_type() == torch::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_F32(x); } while(0)

__global__ void gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K,
    const int TM,
    const int TN
) {
    const int m_start = blockIdx.y * TM;
    const int n_start = blockIdx.x * TN;
    const int actual_tm = min(TM, M - m_start);
    const int actual_tn = min(TN, N - n_start);

    extern __shared__ float sdata[];
    float* shA = sdata;
    float* shB = shA + TM * K;

    const int blockDimX = 16;
    const int tid = threadIdx.y * blockDimX + threadIdx.x;
    const int nthreads = blockDimX * 64;  // 1024

    // Load shA
    for (int ii = tid; ii < TM * K; ii += nthreads) {
        const int row = ii / K;
        const int col = ii % K;
        if (row < actual_tm) {
            shA[row * K + col] = A[(m_start + row) * K + col];
        } else {
            shA[row * K + col] = 0.0f;
        }
    }

    // Load shB
    for (int ii = tid; ii < K * TN; ii += nthreads) {
        const int kk = ii / TN;
        const int col = ii % TN;
        if (n_start + col < N) {
            shB[kk * TN + col] = B[kk * N + n_start + col];
        } else {
            shB[kk * TN + col] = 0.0f;
        }
    }

    __syncthreads();

    // Compute
    const int mm = threadIdx.y * 4;
    const int nn = threadIdx.x * 16;
    float acc[4][16];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int kk = 0; kk < K; ++kk) {
        float avec[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = mm + i;
            avec[i] = shA[row * K + kk];
        }
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int col = nn + j;
            const float bval = shB[kk * TN + col];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[i][j] += avec[i] * bval;
            }
        }
    }

    // Write back
    for (int i = 0; i < 4; ++i) {
        const int g_m = m_start + mm + i;
        if (g_m < M) {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int g_n = n_start + nn + j;
                if (g_n < N) {
                    C[g_m * N + g_n] = acc[i][j];
                }
            }
        }
    }
}

torch::Tensor custom_matmul_hip(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    const auto M_ = A.size(0);
    const auto K_ = A.size(1);
    const auto N_ = B.size(1);

    const int M = static_cast<int>(M_);
    const int K = static_cast<int>(K_);
    const int N = static_cast<int>(N_);

    const int TM = 256;
    const int TN = 256;

    auto C = torch::empty({M_, N_}, A.options());

    const unsigned int grid_x = static_cast<unsigned int>((N + TN - 1) / TN);
    const unsigned int grid_y = static_cast<unsigned int>((M + TM - 1) / TM);
    const dim3 blocks(grid_x, grid_y);
    const dim3 threads(16, 64);

    const size_t shmem_bytes = (static_cast<size_t>(TM) * K + static_cast<size_t>(K) * TN) * sizeof(float);

    hipLaunchKernelGGL(
        gemm_kernel,
        blocks,
        threads,
        shmem_bytes,
        0,
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M,
        N,
        K,
        TM,
        TN
    );

    TORCH_CHECK(hipGetLastError() == hipSuccess, "HIP kernel launch failed");

    return C;
}
"""

matmul = load_inline(
    name="matmul",
    cpp_sources=matmul_cpp_source,
    functions=["custom_matmul_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matmul = matmul

    def forward(self, A, B):
        return self.matmul.custom_matmul_hip(A, B)
