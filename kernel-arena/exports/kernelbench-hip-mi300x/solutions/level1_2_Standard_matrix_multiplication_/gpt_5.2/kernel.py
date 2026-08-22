import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Ensure we compile with HIP on ROCm
os.environ.setdefault("CXX", "hipcc")

# A reasonably optimized FP32 GEMM kernel specialized for row-major contiguous matrices.
# Tile sizes: BM=64, BN=64, BK=16. Threads: (16,16)=256 threads/block.
# Each thread computes a 4x4 micro-tile.

hip_src = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA/HIP tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FP32(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")

static constexpr int BM = 64;
static constexpr int BN = 64;
static constexpr int BK = 16;
static constexpr int TM = 4;
static constexpr int TN = 4;

__global__ void __launch_bounds__(256, 2)
gemm64x64x16_fp32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // Block tile origin
    const int block_m = (int)blockIdx.y * BM;
    const int block_n = (int)blockIdx.x * BN;

    // Thread indices within 16x16
    const int tx = (int)threadIdx.x; // 0..15
    const int ty = (int)threadIdx.y; // 0..15
    const int tid = ty * 16 + tx;    // 0..255

    // Shared memory tiles
    __shared__ float As[BM][BK]; // 64x16
    __shared__ float Bs[BK][BN]; // 16x64

    // Per-thread accumulator for a 4x4 tile
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    // Global row/col base for this thread's micro-tile
    const int row_base = block_m + ty * TM;
    const int col_base = block_n + tx * TN;

    // Loop over K dimension in BK chunks
    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load A tile (64x16) and B tile (16x64) into shared memory.
        // Each thread loads 4 floats from A and 4 floats from B (using float4).

        // A load: 1024 floats total. tid*4 covers 0..1023.
        int a_linear = tid * 4;
        int a_row = a_linear / BK;      // 0..63
        int a_col = a_linear - a_row*BK; // 0,4,8,12

        int a_g_row = block_m + a_row;
        int a_g_col = k0 + a_col;

        float4 a4;
        if (a_g_row < M && (a_g_col + 3) < K) {
            const float4* Ap4 = reinterpret_cast<const float4*>(A + a_g_row * K + a_g_col);
            a4 = *Ap4;
        } else {
            // Tail-safe load
            float tmp[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int gc = a_g_col + i;
                tmp[i] = (a_g_row < M && gc < K) ? A[a_g_row * K + gc] : 0.0f;
            }
            a4 = make_float4(tmp[0], tmp[1], tmp[2], tmp[3]);
        }
        // Store to shared
        *reinterpret_cast<float4*>(&As[a_row][a_col]) = a4;

        // B load: 1024 floats total.
        int b_linear = tid * 4;
        int b_row = b_linear / BN;       // 0..15
        int b_col = b_linear - b_row*BN; // 0,4,8,...,60

        int b_g_row = k0 + b_row;
        int b_g_col = block_n + b_col;

        float4 b4;
        if (b_g_row < K && (b_g_col + 3) < N) {
            const float4* Bp4 = reinterpret_cast<const float4*>(B + b_g_row * N + b_g_col);
            b4 = *Bp4;
        } else {
            float tmp[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int gc = b_g_col + i;
                tmp[i] = (b_g_row < K && gc < N) ? B[b_g_row * N + gc] : 0.0f;
            }
            b4 = make_float4(tmp[0], tmp[1], tmp[2], tmp[3]);
        }
        *reinterpret_cast<float4*>(&Bs[b_row][b_col]) = b4;

        __syncthreads();

        // Compute micro-tile
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float bval[TN];
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int c = tx * TN + j;
                bval[j] = Bs[kk][c];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                int r = ty * TM + i;
                float aval = As[r][kk];
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = fmaf(aval, bval[j], acc[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // Store results
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int r = row_base + i;
        if (r < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int c = col_base + j;
                if (c < N) {
                    C[r * N + c] = acc[i][j];
                }
            }
        }
    }
}

torch::Tensor matmul_hip(torch::Tensor A, torch::Tensor B) {
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(B);
    CHECK_FP32(A);
    CHECK_FP32(B);
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "K dimension mismatch");

    const auto M = (int)A.size(0);
    const auto K = (int)A.size(1);
    const auto N = (int)B.size(1);

    c10::cuda::CUDAGuard device_guard(A.device());
    auto C = torch::empty({M, N}, A.options());

    dim3 block(16, 16, 1);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);

    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    hipLaunchKernelGGL(
        gemm64x64x16_fp32_kernel,
        grid,
        block,
        0,
        stream,
        (const float*)A.data_ptr<float>(),
        (const float*)B.data_ptr<float>(),
        (float*)C.data_ptr<float>(),
        M, N, K
    );

    return C;
}
"""

# Build extension (cached by name)
matmul_ext = load_inline(
    name="matmul_hip_ext_v1",
    cpp_sources="",
    cuda_sources=hip_src,
    functions=["matmul_hip"],
    with_cuda=True,
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_ext = matmul_ext

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Expect A,B on GPU and contiguous FP32
        if not A.is_cuda or not B.is_cuda:
            # Fallback for safety (KernelBench will move to GPU for GPU targets).
            return torch.matmul(A, B)
        if A.dtype != torch.float32:
            A = A.float()
        if B.dtype != torch.float32:
            B = B.float()
        if not A.is_contiguous():
            A = A.contiguous()
        if not B.is_contiguous():
            B = B.contiguous()
        return self.matmul_ext.matmul_hip(A, B)


# Problem sizes (from reference)
M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]


def get_init_inputs():
    return []
