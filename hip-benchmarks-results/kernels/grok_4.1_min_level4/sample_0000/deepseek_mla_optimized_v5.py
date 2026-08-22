import os
os.environ["CXX"] = "hipcc"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

rmsnorm_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void rmsnorm_reduce_var_kernel(const float *input, float *var, int N, int D, float eps) {
  int n = blockIdx.x;
  if (n >= N) return;
  int offset = n * D;
  int tx = threadIdx.x;
  int tile_size = blockDim.x;
  __shared__ float sdata[256];
  __shared__ float accumulator;

  if (tx == 0) accumulator = 0.0f;
  __syncthreads();

  int num_tiles = (D + tile_size - 1) / tile_size;
  for (int tile = 0; tile < num_tiles; ++tile) {
    int col = tile * tile_size + tx;
    float val = (col < D) ? input[offset + col] : 0.0f;
    sdata[tx] = val * val;
    __syncthreads();

    for (int s = tile_size / 2; s > 0; s >>= 1) {
      if (tx < s) sdata[tx] += sdata[tx + s];
      __syncthreads();
    }
    if (tx == 0) accumulator += sdata[0];
    __syncthreads();
  }
  if (tx == 0) var[n] = accumulator / D + eps;
}

__global__ void rmsnorm_apply_kernel(const float *input, const float *var, const float *weight, float *output, int N, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * D) return;
  int n = idx / D;
  int d = idx % D;
  float rvar = 1.0f / sqrtf(var[n]);
  output[idx] = input[idx] * rvar * weight[d];
}

torch::Tensor rmsnorm_hip(torch::Tensor input, torch::Tensor weight, float eps) {
  torch::IntArrayRef in_shape = input.sizes();
  int64_t D = input.size(-1);
  torch::Tensor x = input.reshape(torch::IntArrayRef{-1LL, D}).contiguous().to(torch::kFloat32);
  torch::Tensor w = weight.contiguous().to(torch::kFloat32);
  int64_t N = x.size(0);
  torch::Tensor var_t = torch::empty({N}, x.options());
  const int block_size = 256;
  dim3 block(block_size);
  dim3 grid_var(N);
  size_t shared_var = block_size * sizeof(float) + sizeof(float);
  hipLaunchKernelGGL(rmsnorm_reduce_var_kernel, grid_var, block, shared_var, 0, x.data_ptr<float>(), var_t.data_ptr<float>(), (int)N, (int)D, eps);
  hipDeviceSynchronize();

  torch::Tensor out = torch::empty_like(x);
  dim3 grid_apply((N * D + block_size - 1) / block_size);
  hipLaunchKernelGGL(rmsnorm_apply_kernel, grid_apply, block, 0, 0, x.data_ptr<float>(), var_t.data_ptr<float>(), w.data_ptr<float>(), out.data_ptr<float>(), (int)N, (int)D);
  hipDeviceSynchronize();
  return out.view(in_shape).to(input.dtype());
}
"""

rmsnorm_hip_module = load_inline(
    name="rmsnorm_parallel",
    cpp_sources=rmsnorm_cpp_source,
    functions=["rmsnorm_hip"],
    verbose=True,
)

class DeepSeekRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DeepSeekRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


class ModelNew(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.attention_dropout = attention_dropout
        self.softmax_scale = self.q_head_dim ** (-0.5)

        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_a_layernorm = DeepSeekRMSNorm(q_lora_rank)
        self.q_b_proj = nn.Linear(q_lora_rank, num_attention_heads * self.q_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = DeepSeekRMSNorm(kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            num_attention_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(num_attention_heads * v_head_dim, hidden_size, bias=False)

        self.rotary_emb = DeepSeekRotaryEmbedding(
            qk_rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

        self.rmsnorm_hip = rmsnorm_hip_module

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        q_a = self.q_a_proj(hidden_states)
        q_norm = self.rmsnorm_hip.rmsnorm_hip(q_a, self.q_a_layernorm.weight, self.q_a_layernorm.variance_epsilon)
        q = self.q_b_proj(q_norm)
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)

        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv_full = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe_raw = torch.split(
            compressed_kv_full, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv_norm = self.rmsnorm_hip.rmsnorm_hip(compressed_kv, self.kv_a_layernorm.weight, self.kv_a_layernorm.variance_epsilon)
        kv = self.kv_b_proj(kv_norm)
        kv = kv.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        kv = kv.transpose(1, 2)

        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = k_pe_raw.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin)

        k_pe_expanded = k_pe.expand(bsz, self.num_heads, q_len, self.qk_rope_head_dim)
        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat([k_nope, k_pe_expanded], dim=-1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale

        causal_mask = torch.triu(torch.full((q_len, q_len), float('-inf'), device=hidden_states.device), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output


batch_size = 4
seq_len = 2048
hidden_size = 2048
num_attention_heads = 16
q_lora_rank = 1536
kv_lora_rank = 512
qk_nope_head_dim = 128
qk_rope_head_dim = 64
v_head_dim = 128
max_position_embeddings = 4096


def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size)]


def get_init_inputs():
    return [
        hidden_size,
        num_attention_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        max_position_embeddings,
    ]
