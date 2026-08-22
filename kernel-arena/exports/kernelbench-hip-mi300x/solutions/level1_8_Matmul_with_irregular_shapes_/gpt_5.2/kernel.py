import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Force HIP compilation
os.environ.setdefault("CXX", "hipcc")

# ---------------------------------------------
# HIP GEMM (FP32) tuned for fixed/irregular sizes
# Tile: 64x64x16, 256 threads (16x16), each thread computes 4x4 outputs
# ---------------------------------------------

cpp_source = r"""
#include <torch/extension.h>

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_hip", &matmul_hip, "Custom HIP GEMM FP32 (A[M,K] @ B[K,N])");
}
"""

cuda_source = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <c10/hip/HIPStream.h>

// Simple tiled GEMM kernel: C = A(MxK) * B(KxN)
// Assumes A, B contiguous (row-major).

#ifndef __HIP_PLATFORM_HCC__
#define __HIP_PLATFORM_HCC__ 1
#endif

static inline int div_up(int a, int b) { return (a + b - 1) / b; }

template<int BM, int BN, int BK, int TM, int TN>
__global__ __launch_bounds__(256)
void gemm_tiled_fp32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // 16x16 threads
    const int tx = threadIdx.x; // 0..15
    const int ty = threadIdx.y; // 0..15
    const int tid = ty * 16 + tx;

    const int block_m = (int)blockIdx.y;
    const int block_n = (int)blockIdx.x;

    const int m0 = block_m * BM;
    const int n0 = block_n * BN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load A tile (BM x BK)
        #pragma unroll
        for (int t = 0; t < (BM * BK + 255) / 256; ++t) {
            int idx = tid + t * 256;
            if (idx < BM * BK) {
                int a_r = idx / BK;
                int a_c = idx - a_r * BK;
                int gr = m0 + a_r;
                int gc = k0 + a_c;
                float v = 0.0f;
                if (gr < M && gc < K) {
                    v = A[gr * K + gc];
                }
                As[idx] = v;
            }
        }

        // Load B tile (BK x BN)
        #pragma unroll
        for (int t = 0; t < (BK * BN + 255) / 256; ++t) {
            int idx = tid + t * 256;
            if (idx < BK * BN) {
                int b_r = idx / BN;
                int b_c = idx - b_r * BN;
                int gr = k0 + b_r;
                int gc = n0 + b_c;
                float v = 0.0f;
                if (gr < K && gc < N) {
                    v = B[gr * N + gc];
                }
                Bs[idx] = v;
            }
        }

        __syncthreads();

        // Compute
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float breg[TN];
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int c = tx * TN + j;
                breg[j] = Bs[kk * BN + c];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                int r = ty * TM + i;
                float areg = As[r * BK + kk];
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = fmaf(areg, breg[j], acc[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // Store
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int gr = m0 + ty * TM + i;
        if (gr < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int gc = n0 + tx * TN + j;
                if (gc < N) {
                    C[gr * N + gc] = acc[i][j];
                }
            }
        }
    }
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda(), "A must be on GPU (ROCm/HIP)");
    TORCH_CHECK(B.device().is_cuda(), "B must be on GPU (ROCm/HIP)");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "K dimension mismatch");

    // Ensure contiguous (KernelBench inputs are contiguous already)
    if (!A.is_contiguous()) A = A.contiguous();
    if (!B.is_contiguous()) B = B.contiguous();

    const int M = (int)A.size(0);
    const int K = (int)A.size(1);
    const int N = (int)B.size(1);

    auto C = torch::empty({M, N}, A.options());

    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;
    constexpr int TM = 4;
    constexpr int TN = 4;

    dim3 block(16, 16, 1);
    dim3 grid(div_up(N, BN), div_up(M, BM), 1);

    hipStream_t stream = c10::hip::getCurrentHIPStream();

    const float* Ap = (const float*)A.data_ptr<float>();
    const float* Bp = (const float*)B.data_ptr<float>();
    float* Cp = (float*)C.data_ptr<float>();

    hipLaunchKernelGGL((gemm_tiled_fp32_kernel<BM, BN, BK, TM, TN>),
                       grid, block, 0, stream,
                       Ap, Bp, Cp, M, N, K);

    return C;
}
"""

matmul_ext = load_inline(
    name="matmul_irregular_hip_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=None,
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """Optimized model using a custom HIP GEMM kernel for FP32 matmul."""

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Fallback to PyTorch if on CPU to keep functional parity
        if not A.is_cuda:
            return torch.matmul(A, B)
        return matmul_ext.matmul_hip(A, B)
