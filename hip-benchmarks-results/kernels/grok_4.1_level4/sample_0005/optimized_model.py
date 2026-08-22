import os
os.environ["CXX"] = "hipcc"

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_source = """
#include <hip/hip_runtime.h>

__global__ void dequant_gemm_kernel(
    const half *A, int lda_row,
    const uint8_t *W_packed, int ldwp_row,
    const half *S, int lds_row,
    half *C, int ldc_row,
    int M, int N, int K, int group_size, int num_groups
) {
    constexpr int TM = 64;
    constexpr int TN = 64;
    constexpr int TK = 64;
    constexpr int RS = 4;
    constexpr int CS = 4;
    constexpr int NT_M = TM / RS;
    constexpr int NT_N = TN / CS;
    constexpr int block_size = NT_M * NT_N;
    constexpr int buf_size = TM * TK + TN * TK;

    __shared__ half sh_buf[buf_size];
    half* sh_a = sh_buf;
    half* sh_w = sh_buf + TM * TK;

    int tx = threadIdx.x;
    int lane_m = tx % NT_M;
    int lane_n = tx / NT_M;

    float acc[RS][CS];
    for (int i = 0; i < RS; i++) {
        for (int j = 0; j < CS; j++) {
            acc[i][j] = 0.0f;
        }
    }

    int m_base = blockIdx.x * TM;
    int n_base = blockIdx.y * TN;

    for (int k_outer = 0; k_outer < K; k_outer += TK) {
        // Load sh_a
        for (int i = tx; i < TM * TK; i += block_size) {
            int idx = i;
            int jm = idx / TK;
            int jk = idx % TK;
            int gm = m_base + jm;
            int gk = k_outer + jk;
            if (gm < M && gk < K) {
                sh_a[idx] = A[gm * lda_row + gk];
            } else {
                sh_a[idx] = __float2half(0.0f);
            }
        }

        // Load sh_w
        for (int i = tx; i < TN * TK; i += block_size) {
            int idx = i;
            int jn = idx / TK;
            int jk = idx % TK;
            int gn = n_base + jn;
            int gk = k_outer + jk;
            if (gn < N && gk < K) {
                int byte_idx = gk / 2;
                uint8_t p = W_packed[gn * ldwp_row + byte_idx];
                int wq = (gk % 2 == 0) ? (p & 0x0F) : ((p >> 4) & 0x0F);
                int gg = gk / group_size;
                half sc = S[gn * lds_row + gg];
                half w_h = __float2half(static_cast<float>(wq));
                half centered = w_h - __float2half(8.0f);
                half dw_h = centered * sc;
                sh_w[idx] = dw_h;
            } else {
                sh_w[idx] = __float2half(0.0f);
            }
        }

        __syncthreads();

        // Compute
        for (int tk = 0; tk < TK; ++tk) {
            for (int jj = 0; jj < CS; ++jj) {
                int ln = lane_n * CS + jj;
                if (ln < TN) {
                    float wv = __half2float(sh_w[ln * TK + tk]);
                    for (int ii = 0; ii < RS; ++ii) {
                        int lm = lane_m * RS + ii;
                        if (lm < TM) {
                            float av = __half2float(sh_a[lm * TK + tk]);
                            acc[ii][jj] += av * wv;
                        }
                    }
                }
            }
        }
    }

    // Store
    for (int ii = 0; ii < RS; ++ii) {
        int lm = lane_m * RS + ii;
        int gm = m_base + lm;
        if (gm < M) {
            for (int jj = 0; jj < CS; ++jj) {
                int ln = lane_n * CS + jj;
                int gn = n_base + ln;
                if (gn < N) {
                    C[gm * ldc_row + gn] = __float2half(acc[ii][jj]);
                }
            }
        }
    }
}

torch::Tensor dequant_gemm_forward(torch::Tensor x, torch::Tensor weight_packed, torch::Tensor scales) {
    int M = x.size(0);
    int K = x.size(1);
    int N = weight_packed.size(0);
    int Ng = scales.size(1);
    int group_size = K / Ng;

    auto out = torch::zeros({M, N}, x.options());

    constexpr int TM = 64;
    constexpr int TN = 64;
    constexpr int block_size = 256;
    dim3 block(block_size);
    dim3 grid((M + TM - 1) / TM, (N + TN - 1) / TN);

    const half* a_ptr = static_cast<const half*>(x.data_ptr());
    const uint8_t* wp_ptr = static_cast<const uint8_t*>(weight_packed.data_ptr());
    const half* s_ptr = static_cast<const half*>(scales.data_ptr());
    half* c_ptr = static_cast<half*>(out.data_ptr());

    dequant_gemm_kernel<<<grid, block>>>(
        a_ptr, static_cast<int>(x.stride(0)),
        wp_ptr, static_cast<int>(weight_packed.stride(0)),
        s_ptr, static_cast<int>(scales.stride(0)),
        c_ptr, static_cast<int>(out.stride(0)),
        M, N, K, group_size, Ng
    );

    hipDeviceSynchronize();

    return out;
}
"""

dequant_gemm = load_inline(
    name="dequant_int4_gemm",
    cpp_sources=cpp_source,
    functions=["dequant_gemm_forward"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, K: int, N: int, group_size: int = 128):
        super().__init__()
        self.K = K
        self.N = N
        self.group_size = group_size
        self.num_groups = K // group_size

        assert K % group_size == 0
        assert K % 2 == 0

        self.register_buffer("weight_packed", torch.randint(0, 256, (N, K // 2), dtype=torch.uint8))
        self.register_buffer("scales", torch.randn(N, self.num_groups, dtype=torch.float16).abs() * 0.1)

        self.dequant_gemm = dequant_gemm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x_2d = x.view(-1, self.K)
        out_2d = self.dequant_gemm.dequant_gemm_forward(x_2d, self.weight_packed, self.scales)
        return out_2d.view(batch_size, seq_len, self.N)


batch_size = 4
seq_len = 2048
K = 4096
N = 11008
group_size = 128

def get_inputs():
    return [torch.randn(batch_size, seq_len, K, dtype=torch.float16)]

def get_init_inputs():
    return [K, N, group_size]
