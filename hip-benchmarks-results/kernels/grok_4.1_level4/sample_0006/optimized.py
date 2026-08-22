import os
os.environ["CXX"] = "hipcc"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

gated_delta_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void gated_delta_kernel(
    const float *q, size_t sbhq, size_t ssqq,
    const float *k, size_t sbhk, size_t ssqk,
    const float *v, size_t sbhv, size_t ssvq,
    const float *alpha, size_t sbha, size_t ssaq,
    const float *beta, size_t sbhb, size_t ssbq,
    float *o, size_t sbho, size_t ssqo,
    int bh_total, int n_seq, int dv_loc, int dqk_loc
) {
    int s = blockIdx.x;
    if (s >= bh_total) return;
    extern __shared__ float shmem[];
    float* S_sh = shmem;
    size_t pitch_s = (size_t)dqk_loc;
    int tid = threadIdx.x;
    if (tid < dv_loc) {
        int i = tid;
        size_t ioff = (size_t)i * pitch_s;
        for (int jj = 0; jj < dqk_loc; jj++) {
            S_sh[ioff + jj] = 0.0f;
        }
    }
    __syncthreads();
    for (int t = 0; t < n_seq; t++) {
        size_t base_k = (size_t)s * sbhk + (size_t)t * ssqk;
        size_t base_q = (size_t)s * sbhq + (size_t)t * ssqq;
        size_t base_a = (size_t)s * sbha + (size_t)t * ssaq;
        size_t base_b = (size_t)s * sbhb + (size_t)t * ssbq;
        if (tid < dv_loc) {
            float alph = alpha[base_a];
            float bet = beta[base_b];
            int i = tid;
            size_t ioff = (size_t)i * pitch_s;
            size_t base_v = (size_t)s * sbhv + (size_t)t * ssvq + (size_t)i;
            float vi = v[base_v];
            float sk = 0.0f;
            for (int jj = 0; jj < dqk_loc; jj++) {
                sk += S_sh[ioff + jj] * k[base_k + jj];
            }
            float err = sk - vi;
            for (int jj = 0; jj < dqk_loc; jj++) {
                float sval = S_sh[ioff + jj];
                S_sh[ioff + jj] = alph * sval - bet * err * k[base_k + jj];
            }
            float otv = 0.0f;
            for (int jj = 0; jj < dqk_loc; jj++) {
                otv += S_sh[ioff + jj] * q[base_q + jj];
            }
            size_t base_o = (size_t)s * sbho + (size_t)t * ssqo + (size_t)i;
            o[base_o] = otv;
        }
    }
}

torch::Tensor gated_delta_hip(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor alpha,
    torch::Tensor beta
) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA/HIP tensor");
    TORCH_CHECK(k.is_cuda(), "k must be CUDA/HIP tensor");
    TORCH_CHECK(v.is_cuda(), "v must be CUDA/HIP tensor");
    TORCH_CHECK(alpha.is_cuda(), "alpha must be CUDA/HIP tensor");
    TORCH_CHECK(beta.is_cuda(), "beta must be CUDA/HIP tensor");
    TORCH_CHECK(q.dtype() == torch::kFloat32, "Must be FP32");
    int64_t bh_ = q.size(0);
    int64_t seq_ = q.size(1);
    int64_t dqk_ = q.size(2);
    int64_t dv_ = v.size(2);
    TORCH_CHECK(alpha.size(0) == bh_ && alpha.size(1) == seq_ && alpha.size(2) == 1);
    TORCH_CHECK(beta.size(0) == bh_ && beta.size(1) == seq_ && beta.size(2) == 1);
    TORCH_CHECK(v.size(0) == bh_ && v.size(1) == seq_);
    TORCH_CHECK(k.size(0) == bh_ && k.size(1) == seq_ && k.size(2) == dqk_);
    auto o = torch::empty({bh_, seq_, dv_}, q.options());
    if (bh_ == 0) return o;
    dim3 block(256);
    dim3 grid(static_cast<unsigned int>(bh_));
    size_t shmem_sz = (size_t)dv_ * dqk_ * sizeof(float);
    size_t s_bh_q = q.stride(0);
    size_t s_seq_q = q.stride(1);
    size_t s_bh_k = k.stride(0);
    size_t s_seq_k = k.stride(1);
    size_t s_bh_v = v.stride(0);
    size_t s_seq_v = v.stride(1);
    size_t s_bh_a = alpha.stride(0);
    size_t s_seq_a = alpha.stride(1);
    size_t s_bh_b = beta.stride(0);
    size_t s_seq_b = beta.stride(1);
    size_t s_bh_o = o.stride(0);
    size_t s_seq_o = o.stride(1);
    int i_bh = static_cast<int>(bh_);
    int i_seq = static_cast<int>(seq_);
    int i_dv = static_cast<int>(dv_);
    int i_dqk = static_cast<int>(dqk_);
    hipLaunchKernelGGL(gated_delta_kernel, grid, block, shmem_sz, 0,
        q.data_ptr<float>(), s_bh_q, s_seq_q,
        k.data_ptr<float>(), s_bh_k, s_seq_k,
        v.data_ptr<float>(), s_bh_v, s_seq_v,
        alpha.data_ptr<float>(), s_bh_a, s_seq_a,
        beta.data_ptr<float>(), s_bh_b, s_seq_b,
        o.data_ptr<float>(), s_bh_o, s_seq_o,
        i_bh, i_seq, i_dv, i_dqk);
    return o;
}
"""

gated_delta = load_inline(
    name="gated_delta",
    cpp_sources=gated_delta_cpp_source,
    functions=["gated_delta_hip"],
    verbose=True,
)


# Gated DeltaNet: Linear Attention with Gated Delta Rule
# ... (keep the comment)

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
        self.gated_delta = gated_delta

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)

        # Gating projections
        self.a_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=True)

        # Output projection
        self.o_proj = nn.Linear(num_heads * head_dim_v, hidden_size, bias=False)

        # Optional short convolution for local context
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

        # Output gate with RMSNorm + SiLU
        self.g_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)
        self.o_norm = nn.LayerNorm(head_dim_v)

        # Scaling factor for keys (prevents state explosion)
        self.scale = head_dim_qk ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq, num_heads * head_dim_qk)
        k = self.k_proj(x)  # (batch, seq, num_heads * head_dim_qk)
        v = self.v_proj(x)  # (batch, seq, num_heads * head_dim_v)

        # Optional short convolution
        if self.use_short_conv:
            q = self.q_conv(q.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            k = self.k_conv(k.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            v = self.v_conv(v.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            q = F.silu(q)
            k = F.silu(k)
            v = F.silu(v)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim_qk)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim_qk)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim_v)

        # Compute gating values
        alpha = torch.sigmoid(self.a_proj(x))  # (batch, seq, num_heads)
        beta = torch.sigmoid(self.b_proj(x))   # (batch, seq, num_heads)

        # Scale keys to prevent state explosion
        k = k * self.scale

        # Optimized kernel call
        bh = batch_size * self.num_heads
        q_resh = q.transpose(1, 2).contiguous().view(bh, seq_len, self.head_dim_qk)
        k_resh = k.transpose(1, 2).contiguous().view(bh, seq_len, self.head_dim_qk)
        v_resh = v.transpose(1, 2).contiguous().view(bh, seq_len, self.head_dim_v)
        alpha_resh = alpha.transpose(1, 2).contiguous().view(bh, seq_len, 1)
        beta_resh = beta.transpose(1, 2).contiguous().view(bh, seq_len, 1)

        o_resh = self.gated_delta.gated_delta_hip(q_resh, k_resh, v_resh, alpha_resh, beta_resh)

        # Reshape back
        o = o_resh.view(batch_size, self.num_heads, seq_len, self.head_dim_v).transpose(1, 2)

        # Apply output normalization per head
        o = self.o_norm(o)

        # Apply output gate
        g = torch.sigmoid(self.g_proj(x))  # (batch, seq, num_heads * head_dim_v)
        g = g.view(batch_size, seq_len, self.num_heads, self.head_dim_v)
        o = o * g

        # Reshape and project output
        o = o.reshape(batch_size, seq_len, self.num_heads * self.head_dim_v)
        o = self.o_proj(o)

        return o


# Configuration matching typical LLM settings
batch_size = 4
seq_len = 2048
hidden_size = 2048
num_heads = 16
head_dim_qk = 128  # Key/query dimension per head
head_dim_v = 128   # Value dimension per head


def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size)]


def get_init_inputs():
    return [hidden_size, num_heads, head_dim_qk, head_dim_v]
