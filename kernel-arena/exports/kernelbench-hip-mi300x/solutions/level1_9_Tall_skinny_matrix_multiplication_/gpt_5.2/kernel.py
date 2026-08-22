import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Compile with hipcc for ROCm
os.environ.setdefault("CXX", "hipcc")

# K=32 specialized GEMM: (M x 32) @ (32 x N) -> (M x N)
# Tuned for large M,N and small K (32).

hip_source = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA/HIP tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")

// 64x64 tile, 16x16 threads, each thread computes a 4x4 micro-tile.
// Shared memory uses padding to mitigate bank conflicts.
__global__ void gemm_k32_tiled_64x64_kernel(
    const float* __restrict__ A, // [M, 32]
    const float* __restrict__ B, // [32, N]
    float* __restrict__ C,       // [M, N]
    int M, int N)
{
    // block tile origin
    const int block_row = (int)blockIdx.y * 64;
    const int block_col = (int)blockIdx.x * 64;

    const int tx = (int)threadIdx.x; // 0..15
    const int ty = (int)threadIdx.y; // 0..15
    const int tid = ty * 16 + tx;

    __shared__ float As[64][33];   // [BM][K+1]
    __shared__ float Bs[32][65];   // [K][BN+1]

    // Load A tile: 64x32 = 2048 floats = 512 float4 loads.
    // Each thread loads 2 float4.
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int vec_id = tid * 2 + i;          // 0..511
        int elem = vec_id * 4;            // 0..2044 step 4
        int r = elem >> 5;                // /32, 0..63
        int k = elem & 31;                // %32, multiple of 4
        int gr = block_row + r;
        if (gr < M) {
            const float4 v = *reinterpret_cast<const float4*>(A + gr * 32 + k);
            As[r][k + 0] = v.x;
            As[r][k + 1] = v.y;
            As[r][k + 2] = v.z;
            As[r][k + 3] = v.w;
        } else {
            // out of bounds rows
            As[r][k + 0] = 0.0f;
            As[r][k + 1] = 0.0f;
            As[r][k + 2] = 0.0f;
            As[r][k + 3] = 0.0f;
        }
    }

    // Load B tile: 32x64 = 2048 floats = 512 float4 loads.
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int vec_id = tid * 2 + i;          // 0..511
        int elem = vec_id * 4;
        int k = elem >> 6;                 // /64, 0..31
        int c = elem & 63;                 // %64, multiple of 4
        int gc = block_col + c;
        if (gc + 3 < N) {
            const float4 v = *reinterpret_cast<const float4*>(B + k * N + gc);
            Bs[k][c + 0] = v.x;
            Bs[k][c + 1] = v.y;
            Bs[k][c + 2] = v.z;
            Bs[k][c + 3] = v.w;
        } else {
            // tail (rare for benchmark sizes)
            float tmp0 = 0.f, tmp1 = 0.f, tmp2 = 0.f, tmp3 = 0.f;
            if (gc + 0 < N) tmp0 = B[k * N + (gc + 0)];
            if (gc + 1 < N) tmp1 = B[k * N + (gc + 1)];
            if (gc + 2 < N) tmp2 = B[k * N + (gc + 2)];
            if (gc + 3 < N) tmp3 = B[k * N + (gc + 3)];
            Bs[k][c + 0] = tmp0;
            Bs[k][c + 1] = tmp1;
            Bs[k][c + 2] = tmp2;
            Bs[k][c + 3] = tmp3;
        }
    }

    __syncthreads();

    // Each thread computes C for rows (ty*4..ty*4+3) and cols (tx*4..tx*4+3)
    const int row0 = ty * 4;
    const int col0 = tx * 4;

    float acc00 = 0.f, acc01 = 0.f, acc02 = 0.f, acc03 = 0.f;
    float acc10 = 0.f, acc11 = 0.f, acc12 = 0.f, acc13 = 0.f;
    float acc20 = 0.f, acc21 = 0.f, acc22 = 0.f, acc23 = 0.f;
    float acc30 = 0.f, acc31 = 0.f, acc32 = 0.f, acc33 = 0.f;

    #pragma unroll
    for (int k = 0; k < 32; ++k) {
        float a0 = As[row0 + 0][k];
        float a1 = As[row0 + 1][k];
        float a2 = As[row0 + 2][k];
        float a3 = As[row0 + 3][k];

        float b0 = Bs[k][col0 + 0];
        float b1 = Bs[k][col0 + 1];
        float b2 = Bs[k][col0 + 2];
        float b3 = Bs[k][col0 + 3];

        acc00 = fmaf(a0, b0, acc00);
        acc01 = fmaf(a0, b1, acc01);
        acc02 = fmaf(a0, b2, acc02);
        acc03 = fmaf(a0, b3, acc03);

        acc10 = fmaf(a1, b0, acc10);
        acc11 = fmaf(a1, b1, acc11);
        acc12 = fmaf(a1, b2, acc12);
        acc13 = fmaf(a1, b3, acc13);

        acc20 = fmaf(a2, b0, acc20);
        acc21 = fmaf(a2, b1, acc21);
        acc22 = fmaf(a2, b2, acc22);
        acc23 = fmaf(a2, b3, acc23);

        acc30 = fmaf(a3, b0, acc30);
        acc31 = fmaf(a3, b1, acc31);
        acc32 = fmaf(a3, b2, acc32);
        acc33 = fmaf(a3, b3, acc33);
    }

    // Store results
    const int gr = block_row + row0;
    const int gc = block_col + col0;

    if (gr + 3 < M && gc + 3 < N) {
        // vectorized stores
        *reinterpret_cast<float4*>(C + (gr + 0) * N + gc) = make_float4(acc00, acc01, acc02, acc03);
        *reinterpret_cast<float4*>(C + (gr + 1) * N + gc) = make_float4(acc10, acc11, acc12, acc13);
        *reinterpret_cast<float4*>(C + (gr + 2) * N + gc) = make_float4(acc20, acc21, acc22, acc23);
        *reinterpret_cast<float4*>(C + (gr + 3) * N + gc) = make_float4(acc30, acc31, acc32, acc33);
    } else {
        // tails
        if (gr + 0 < M) {
            if (gc + 0 < N) C[(gr + 0) * N + (gc + 0)] = acc00;
            if (gc + 1 < N) C[(gr + 0) * N + (gc + 1)] = acc01;
            if (gc + 2 < N) C[(gr + 0) * N + (gc + 2)] = acc02;
            if (gc + 3 < N) C[(gr + 0) * N + (gc + 3)] = acc03;
        }
        if (gr + 1 < M) {
            if (gc + 0 < N) C[(gr + 1) * N + (gc + 0)] = acc10;
            if (gc + 1 < N) C[(gr + 1) * N + (gc + 1)] = acc11;
            if (gc + 2 < N) C[(gr + 1) * N + (gc + 2)] = acc12;
            if (gc + 3 < N) C[(gr + 1) * N + (gc + 3)] = acc13;
        }
        if (gr + 2 < M) {
            if (gc + 0 < N) C[(gr + 2) * N + (gc + 0)] = acc20;
            if (gc + 1 < N) C[(gr + 2) * N + (gc + 1)] = acc21;
            if (gc + 2 < N) C[(gr + 2) * N + (gc + 2)] = acc22;
            if (gc + 3 < N) C[(gr + 2) * N + (gc + 3)] = acc23;
        }
        if (gr + 3 < M) {
            if (gc + 0 < N) C[(gr + 3) * N + (gc + 0)] = acc30;
            if (gc + 1 < N) C[(gr + 3) * N + (gc + 1)] = acc31;
            if (gc + 2 < N) C[(gr + 3) * N + (gc + 2)] = acc32;
            if (gc + 3 < N) C[(gr + 3) * N + (gc + 3)] = acc33;
        }
    }
}

torch::Tensor matmul_k32_hip(torch::Tensor A, torch::Tensor B) {
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    CHECK_FLOAT(A);
    CHECK_FLOAT(B);

    // Expect 2D
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t K2 = B.size(0);
    int64_t N = B.size(1);
    TORCH_CHECK(K == K2, "inner dimensions mismatch");

    // Specialize for K=32 and contiguous
    if (K != 32 || !A.is_contiguous() || !B.is_contiguous()) {
        return at::matmul(A, B);
    }

    auto C = torch::empty({M, N}, A.options());

    const dim3 block(16, 16, 1);
    const dim3 grid((unsigned int)((N + 63) / 64), (unsigned int)((M + 63) / 64), 1);

    hipStream_t stream = (hipStream_t)at::cuda::getDefaultCUDAStream();
    hipLaunchKernelGGL(
        gemm_k32_tiled_64x64_kernel,
        grid,
        block,
        0,
        stream,
        (const float*)A.data_ptr<float>(),
        (const float*)B.data_ptr<float>(),
        (float*)C.data_ptr<float>(),
        (int)M,
        (int)N
    );
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_k32_hip", &matmul_k32_hip, "K=32 matmul (HIP)");
}
'''

# Build extension (cached by name)
matmul_k32_ext = load_inline(
    name="matmul_k32_ext",
    cpp_sources=hip_source,
    functions=None,
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """Optimized Model using a custom HIP kernel specialized for K=32."""

    def __init__(self):
        super().__init__()
        self.ext = matmul_k32_ext

    def forward(self, A, B):
        # Fallback for CPU tensors
        if not A.is_cuda or not B.is_cuda:
            return torch.matmul(A, B)
        return self.ext.matmul_k32_hip(A, B)
