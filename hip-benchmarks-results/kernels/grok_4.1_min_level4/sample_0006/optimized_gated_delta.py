import os
os.environ["CXX"] = "hipcc"
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

gated_delta_cpp_source = """
#include <hip/hip_runtime.h>

constexpr int D = 128;
__global__ void gated_delta_kernel(const float *q_ptr, const float *k_ptr, const float *v_ptr, const float *alpha_ptr, const float *beta_ptr, float *o_ptr, float *s_ptr, int bh_stride, int s_bh_stride, int o_bh_stride, int Seq) {
  __shared__ float qt_shared[D];
  __shared__ float kt_shared[D];
  __shared__ float vt_shared[D];
  __shared__ float alpha_shared;
  __shared__ float beta_shared;

  int bh = blockIdx.x;
  int i = threadIdx.x;
  if (i >= D) return;

  int bh_offset = bh * bh_stride;
  int s_bh_offset = bh * s_bh_stride;
  int o_bh_offset = bh * o_bh_stride;

  for (int t = 0; t < Seq; t++) {
    qt_shared[i] = q_ptr[bh_offset + t * D + i];
    kt_shared[i] = k_ptr[bh_offset + t * D + i];
    vt_shared[i] = v_ptr[bh_offset + t * D + i];
    __syncthreads();
    if (i == 0) {
      alpha_shared = alpha_ptr[bh * Seq + t];
      beta_shared = beta_ptr[bh * Seq + t];
    }
    __syncthreads();

    float sk = 0.0f;
    for (int j = 0; j < D; j++) {
      sk += s_ptr[s_bh_offset + i * D + j] * kt_shared[j];
    }
    float err = sk - vt_shared[i];

    for (int j = 0; j < D; j++) {
      float sj = s_ptr[s_bh_offset + i * D + j];
      s_ptr[s_bh_offset + i * D + j] = alpha_shared * sj - beta_shared * err * kt_shared[j];
    }
    __syncthreads();

    float ot = 0.0f;
    for (int j = 0; j < D; j++) {
      ot += s_ptr[s_bh_offset + i * D + j] * qt_shared[j];
    }
    o_ptr[o_bh_offset + t * D + i] = ot;
  }
}

torch::Tensor gated_delta_hip(torch::Tensor q_resh, torch::Tensor k_resh, torch::Tensor v_resh, torch::Tensor alpha_resh, torch::Tensor beta_resh) {
  int64_t BH = q_resh.size(0);
  int64_t Seq = q_resh.size(1);
  auto options = q_resh.options();
  auto s = torch::zeros({BH, (int64_t)D, (int64_t)D}, options);
  auto o = torch::empty({BH, Seq, (int64_t)D}, options);
  dim3 block(D);
  dim3 grid((unsigned int)BH);
  int bh_stride = (int)(Seq * D);
  int s_bh_stride = D * D;
  int o_bh_stride = (int)(Seq * D);
  gated_delta_kernel<<<grid, block>>>(q_resh.data_ptr<float>(), k_resh.data_ptr<float>(), v_resh.data_ptr<float>(), alpha_resh.data_ptr<float>(), beta_resh.data_ptr<float>(), o.data_ptr<float>(), s.data_ptr<float>(), bh_stride, s_bh_stride, o_bh_stride, (int)Seq);
  hipDeviceSynchronize();
  return o;
}
"""

gated_delta = load_inline(
    name="gated_delta",
    cpp_sources=gated_delta_cpp_source,
    functions=["gated_delta_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim_qk: int,
        head_dim_v: int,
        use_short_conv: bool = True,
        conv_kernel_size: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_v
        self.use_short_conv = use_short_conv
        self.conv_kernel_size = conv_kernel_size

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)

        self.a_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=True)

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

        self.gated_delta = gated_delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.use_short_conv:
            q = self.q_conv(q.transpose(1, 2))[:, :, :S].transpose(1, 2)
            k = self.k_conv(k.transpose(1, 2))[:, :, :S].transpose(1, 2)
            v = self.v_conv(v.transpose(1, 2))[:, :, :S].transpose(1, 2)
            q = F.silu(q)
            k = F.silu(k)
            v = F.silu(v)

        q = q.view(B, S, self.num_heads, self.head_dim_qk)
        k = k.view(B, S, self.num_heads, self.head_dim_qk)
        v = v.view(B, S, self.num_heads, self.head_dim_v)

        alpha = torch.sigmoid(self.a_proj(x))
        beta = torch.sigmoid(self.b_proj(x))

        k = k * self.scale

        BH = B * self.num_heads
        q_resh = q.permute(0, 2, 1, 3).contiguous().view(BH, S, self.head_dim_qk)
        k_resh = k.permute(0, 2, 1, 3).contiguous().view(BH, S, self.head_dim_qk)
        v_resh = v.permute(0, 2, 1, 3).contiguous().view(BH, S, self.head_dim_v)
        alpha_resh = alpha.transpose(1, 2).contiguous().view(BH, S)
        beta_resh = beta.transpose(1, 2).contiguous().view(BH, S)

        o_heads = self.gated_delta.gated_delta_hip(q_resh, k_resh, v_resh, alpha_resh, beta_resh)

        o = o_heads.view(B, self.num_heads, S, self.head_dim_v).permute(0, 2, 1, 3)

        o = self.o_norm(o)

        g = torch.sigmoid(self.g_proj(x)).view(B, S, self.num_heads, self.head_dim_v)
        o = o * g

        o = o.reshape(B, S, self.num_heads * self.head_dim_v)
        o = self.o_proj(o)

        return o

batch_size = 4
seq_len = 2048
hidden_size = 2048
num_heads = 16
head_dim_qk = 128
head_dim_v = 128

def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size).cuda()]

def get_init_inputs():
    return [hidden_size, num_heads, head_dim_qk, head_dim_v]
