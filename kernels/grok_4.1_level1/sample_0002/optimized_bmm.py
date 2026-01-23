import os
os.environ["CXX"] = "hipcc"
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

bmm_cpp = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>

#define BLOCK_M 64
#define BLOCK_N 256
#define BK 32
#define WM 4
#define WN 16

__shared__ float a_smem[BLOCK_M][BK];
__shared__ float b_smem[BK][BLOCK_N];

__global__ void tiled_bmm_kernel(
    const float *A, 
    const float *B, 
    float *C,
    int bs, 
    int M, 
    int N, 
    int K
) {
    int b = blockIdx.z;
    if (b >= bs) return;

    const float *a = A + static_cast<size_t>(b) * M * K;
    const float *bb = B + static_cast<size_t>(b) * K * N;
    float *c = C + static_cast<size_t>(b) * M * N;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float acc[WM][WN] = {0.0f};

    int num_tiles_k = (K + BK - 1) / BK;
    for (int kt = 0; kt < num_tiles_k; ++kt) {
        int k_start = kt * BK;

        // Load A tile
        int idx_a = ty * 16 + tx;
        while (idx_a < BLOCK_M * BK) {
            int row_a = idx_a / BK;
            int col_ka = idx_a % BK;
            int g_row_a = bx * BLOCK_M + row_a;
            if (g_row_a < M && k_start + col_ka < K) {
                a_smem[row_a][col_ka] = a[g_row_a * K + k_start + col_ka];
            } else {
                a_smem[row_a][col_ka] = 0.0f;
            }
            idx_a += 256;
        }

        // Load B tile
        int idx_b = ty * 16 + tx;
        while (idx_b < BK * BLOCK_N) {
            int row_kb = idx_b / BLOCK_N;
            int col_nb = idx_b % BLOCK_N;
            int g_col_b = by * BLOCK_N + col_nb;
            if (k_start + row_kb < K && g_col_b < N) {
                b_smem[row_kb][col_nb] = bb[(k_start + row_kb) * N + g_col_b];
            } else {
                b_smem[row_kb][col_nb] = 0.0f;
            }
            idx_b += 256;
        }

        __syncthreads();

        // Compute
#pragma unroll
        for (int k_tile_local = 0; k_tile_local < BK; ++k_tile_local) {
#pragma unroll
            for (int wm = 0; wm < WM; ++wm) {
                float a_val = a_smem[tx * WM + wm][k_tile_local];
#pragma unroll
                for (int wn = 0; wn < WN; ++wn) {
                    float b_val = b_smem[k_tile_local][ty * WN + wn];
                    acc[wm][wn] += a_val * b_val;
                }
            }
        }

        __syncthreads();
    }

    // Write back C
    int row_start = bx * BLOCK_M + tx * WM;
    int col_start = by * BLOCK_N + ty * WN;
    if (row_start < M) {
#pragma unroll
        for (int wm = 0; wm < WM; ++wm) {
            int row = row_start + wm;
            if (row >= M) break;
#pragma unroll
            for (int wn = 0; wn < WN; ++wn) {
                int col = col_start + wn;
                if (col >= N) break;
                c[row * N + col] = acc[wm][wn];
            }
        }
    }
}

torch::Tensor tiled_bmm_hip(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be CUDA tensor");
    TORCH_CHECK(A.scalar_type() == at::ScalarType::Float, "A must be float32");
    TORCH_CHECK(B.scalar_type() == at::ScalarType::Float, "B must be float32");
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3, "Expect 3D tensors");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner matrix dimensions must match");

    int64_t bs_ = A.size(0);
    int64_t M_ = A.size(1);
    int64_t K_ = A.size(2);
    int64_t N_ = B.size(2);
    int bs = static_cast<int>(bs_);
    int M = static_cast<int>(M_);
    int K = static_cast<int>(K_);
    int N = static_cast<int>(N_);

    auto C = torch::zeros({bs_, M_, N_}, A.options());

    dim3 threads(16, 16);
    dim3 blocks(
        (M + BLOCK_M - 1) / BLOCK_M,
        (N + BLOCK_N - 1) / BLOCK_N,
        static_cast<unsigned int>(bs)
    );
    size_t shmem_bytes = sizeof(float) * (BLOCK_M * BK + BK * BLOCK_N);

    tiled_bmm_kernel<<<blocks, threads, shmem_bytes>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        bs,
        M,
        N,
        K
    );

    return C;
}
"""

bmm_module = load_inline(
    name="tiled_bmm",
    cpp_sources=bmm_cpp,
    functions=["tiled_bmm_hip"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.bmm_module = bmm_module

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.bmm_module.tiled_bmm_hip(A, B)
