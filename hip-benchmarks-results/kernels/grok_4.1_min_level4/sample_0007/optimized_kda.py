import os
os.environ["CXX"] = "hipcc"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

kda_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void kda_kernel(
    const float* __restrict__ q_ptr,
    const float* __restrict__ k_ptr,
    const float* __restrict__ v_ptr,
    const float* __restrict__ a_ptr,
    const float* __restrict__ beta_ptr,
    float* __restrict__ o_ptr,
    const int64_t N,
    const int64_t S,
    const int64_t n_stride_kq,
    const int64_t s_stride_kq,
    const int64_t n_stride_va,
    const int64_t s_stride_va,
    const int64_t s_stride_beta
) {
  const int n = blockIdx.x;
  if (n >= N) return;

  const int64_t base_k = n * n_stride_kq;
  const int64_t base_q = base_k;
  const int64_t base_v = n * n_stride_va;
  const int64_t base_a = base_v;
  const int64_t base_beta = n * S * s_stride_beta;
  const int64_t base_o = base_v;

  constexpr int DV_ = 128;
  constexpr int DQ_ = 128;

  __shared__ float S_shared[DV_][DQ_];

  const int row = threadIdx.x;
  if (row >= DV_) return;

  // Initialize S to zero
  for (int j = 0; j < DQ_; ++j) {
    S_shared[row][j] = 0.0f;
  }
  __syncthreads();

  for (int64_t t = 0; t < S; ++t) {
    const int64_t kt_offset = t * s_stride_kq;
    const int64_t qt_offset = t * s_stride_kq;
    const int64_t vt_offset = t * s_stride_va;
    const int64_t at_offset = t * s_stride_va;
    const int64_t ot_offset = t * s_stride_va;

    const float betat = beta_ptr[base_beta + t * s_stride_beta];
    const float a_row = a_ptr[base_a + at_offset + row];
    const float v_row = v_ptr[base_v + vt_offset + row];

    // Compute sk = S[row] @ k_t
    float sk_reg = 0.0f;
    for (int j = 0; j < DQ_; ++j) {
      sk_reg = __fmaf_rn(S_shared[row][j], k_ptr[base_k + kt_offset + j], sk_reg);
    }

    const float error_reg = sk_reg - v_row;

    // Update this row
    for (int j = 0; j < DQ_; ++j) {
      float val = S_shared[row][j] * a_row;
      val -= betat * error_reg * k_ptr[base_k + kt_offset + j];
      S_shared[row][j] = val;
    }
    __syncthreads();

    // Compute o_t[row] = S[row] @ q_t
    float o_reg = 0.0f;
    for (int j = 0; j < DQ_; ++j) {
      o_reg = __fmaf_rn(S_shared[row][j], q_ptr[base_q + qt_offset + j], o_reg);
    }
    o_ptr[base_o + ot_offset + row] = o_reg;
  }
}

torch::Tensor kda_hip(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor beta) {
  auto q_sizes = q.sizes();
  int64_t N = q_sizes[0];
  int64_t S = q_sizes[1];
  int64_t DQ = q_sizes[2];

  auto v_sizes = v.sizes();
  int64_t DV = v_sizes[2];

  torch::Tensor o = torch::empty({N, S, DV}, q.options());

  int64_t n_stride_kq = S * DQ;
  int64_t s_stride_kq = DQ;
  int64_t n_stride_va = S * DV;
  int64_t s_stride_va = DV;
  int64_t s_stride_beta = 1;

  const int block_size = 128;
  dim3 grid(N);
  dim3 block(block_size);

  kda_kernel<<<grid, block>>>(
      q.data_ptr<float>(),
      k.data_ptr<float>(),
      v.data_ptr<float>(),
      a.data_ptr<float>(),
      beta.data_ptr<float>(),
      o.data_ptr<float>(),
      N, S,
      n_stride_kq, s_stride_kq,
      n_stride_va, s_stride_va,
      s_stride_beta
  );

  (void) hipDeviceSynchronize();

  return o;
}
"""

kda = load_inline(
    name="kda",
    cpp_sources=kda_cpp_source,
    functions=["kda_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim_qk: int,
        head_dim_v: int,
        use_dplr: bool = False,
        dplr_rank: int = 4,
        use_short_conv: bool = True,
        conv_kernel_size: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_v
        self.use_dplr = use_dplr
        self.dplr_rank = dplr_rank
        self.use_short_conv = use_short_conv

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)

        self.a_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=True)

        self.b_proj = nn.Linear(hidden_size, num_heads, bias=True)

        if use_dplr:
            self.l_proj = nn.Linear(hidden_size, num_heads * dplr_rank, bias=False)
            self.r_proj = nn.Linear(hidden_size, num_heads * dplr_rank, bias=False)

        self.o_proj = nn.Linear(num_heads * head_dim_v, hidden_size, bias=False)

        if use_short_conv:
            self.q_conv = nn.Conv1d(
                num_heads * head_dim_qk, num_heads * head_dim_qk,
                kernel_size=conv_kernel_size, groups=num_heads * head_dim_qk,
                padding=conv_kernel_size - 1
            )
            self.k_conv = nn.Conv1d(
                num_heads * head_dim_qk, num_heads * head_dim_qk,
                kernel_size=conv_kernel_size, groups=num_heads * head_dim_qk,
                padding=conv_kernel_size - 1
            )
            self.v_conv = nn.Conv1d(
                num_heads * head_dim_v, num_heads * head_dim_v,
                kernel_size=conv_kernel_size, groups=num_heads * head_dim_v,
                padding=conv_kernel_size - 1
            )

        self.g_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)
        self.o_norm = nn.LayerNorm(head_dim_v)

        self.scale = head_dim_qk ** -0.5

        self.kda = kda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.use_short_conv:
            q = self.q_conv(q.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            k = self.k_conv(k.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            v = self.v_conv(v.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            q = F.silu(q)
            k = F.silu(k)
            v = F.silu(v)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim_qk)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim_qk)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim_v)

        a = torch.sigmoid(self.a_proj(x))
        a = a.view(batch_size, seq_len, self.num_heads, self.head_dim_v)

        beta = torch.sigmoid(self.b_proj(x))
        beta = beta.unsqueeze(-1)

        if self.use_dplr:
            l = self.l_proj(x).view(batch_size, seq_len, self.num_heads, self.dplr_rank)
            r = self.r_proj(x).view(batch_size, seq_len, self.num_heads, self.dplr_rank)

        k = k * self.scale

        N = batch_size * self.num_heads
        q_resh = q.permute(0, 2, 1, 3).contiguous().view(N, seq_len, self.head_dim_qk)
        k_resh = k.permute(0, 2, 1, 3).contiguous().view(N, seq_len, self.head_dim_qk)
        v_resh = v.permute(0, 2, 1, 3).contiguous().view(N, seq_len, self.head_dim_v)
        a_resh = a.permute(0, 2, 1, 3).contiguous().view(N, seq_len, self.head_dim_v)
        beta_resh = beta.view(N, seq_len, 1).contiguous()

        o_flat = self.kda.kda_hip(q_resh, k_resh, v_resh, a_resh, beta_resh)

        o = o_flat.view(batch_size, self.num_heads, seq_len, self.head_dim_v).permute(0, 2, 1, 3)

        o = self.o_norm(o)

        g = torch.sigmoid(self.g_proj(x))
        g = g.view(batch_size, seq_len, self.num_heads, self.head_dim_v)
        o = o * g

        o = o.reshape(batch_size, seq_len, self.num_heads * self.head_dim_v)
        o = self.o_proj(o)

        return o


batch_size = 4
seq_len = 2048
hidden_size = 2048
num_heads = 16
head_dim_qk = 128
head_dim_v = 128


def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size)]


def get_init_inputs():
    return [hidden_size, num_heads, head_dim_qk, head_dim_v]
