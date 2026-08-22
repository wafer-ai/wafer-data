import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused attention kernel with flash attention style
attention_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <float.h>

// Block sizes for attention
#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 32

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

__global__ void masked_softmax_kernel(
    float* __restrict__ attn_weights,
    int seq_len,
    int num_heads,
    int batch_size
) {
    // Each block handles one row of attention for one head
    int batch_head = blockIdx.x;
    int row = blockIdx.y;
    
    int total_heads = batch_size * num_heads;
    if (batch_head >= total_heads || row >= seq_len) return;
    
    float* row_ptr = attn_weights + (long long)batch_head * seq_len * seq_len + (long long)row * seq_len;
    
    // Causal masking: only attend to positions <= row
    int valid_len = row + 1;
    
    // Find max (for numerical stability)
    float max_val = -FLT_MAX;
    for (int j = threadIdx.x; j < valid_len; j += blockDim.x) {
        max_val = fmaxf(max_val, row_ptr[j]);
    }
    
    // Warp reduction for max
    max_val = warp_reduce_max(max_val);
    
    // Block reduction for max
    __shared__ float shared_max[8];
    int warp_id = threadIdx.x / 64;
    int lane = threadIdx.x % 64;
    if (lane == 0) shared_max[warp_id] = max_val;
    __syncthreads();
    
    if (threadIdx.x == 0) {
        float m = -FLT_MAX;
        int nwarps = (blockDim.x + 63) / 64;
        for (int i = 0; i < nwarps; i++) m = fmaxf(m, shared_max[i]);
        shared_max[0] = m;
    }
    __syncthreads();
    max_val = shared_max[0];
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int j = threadIdx.x; j < valid_len; j += blockDim.x) {
        float exp_val = expf(row_ptr[j] - max_val);
        row_ptr[j] = exp_val;
        sum += exp_val;
    }
    
    // Set masked positions to 0
    for (int j = valid_len + threadIdx.x; j < seq_len; j += blockDim.x) {
        row_ptr[j] = 0.0f;
    }
    
    // Warp reduction for sum
    sum = warp_reduce_sum(sum);
    
    // Block reduction for sum
    __shared__ float shared_sum[8];
    if (lane == 0) shared_sum[warp_id] = sum;
    __syncthreads();
    
    if (threadIdx.x == 0) {
        float s = 0.0f;
        int nwarps = (blockDim.x + 63) / 64;
        for (int i = 0; i < nwarps; i++) s += shared_sum[i];
        shared_sum[0] = 1.0f / s;
    }
    __syncthreads();
    float inv_sum = shared_sum[0];
    
    // Normalize
    for (int j = threadIdx.x; j < valid_len; j += blockDim.x) {
        row_ptr[j] *= inv_sum;
    }
}

void masked_softmax_hip(torch::Tensor attn_weights, int seq_len, int num_heads, int batch_size) {
    int total_heads = batch_size * num_heads;
    
    dim3 blocks(total_heads, seq_len);
    int threads = 256;
    
    masked_softmax_kernel<<<blocks, threads>>>(
        attn_weights.data_ptr<float>(),
        seq_len,
        num_heads,
        batch_size
    );
}
"""

attention_module = load_inline(
    name="attention_hip_v6",
    cpp_sources="void masked_softmax_hip(torch::Tensor attn_weights, int seq_len, int num_heads, int batch_size);",
    cuda_sources=attention_source,
    functions=["masked_softmax_hip"],
    verbose=False,
    extra_cuda_cflags=["-O3"]
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

        self.kv_a_proj_with_mqa = nn.Linear(hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = DeepSeekRMSNorm(kv_lora_rank)
        self.kv_b_proj = nn.Linear(kv_lora_rank, num_attention_heads * (qk_nope_head_dim + v_head_dim), bias=False)

        self.o_proj = nn.Linear(num_attention_heads * v_head_dim, hidden_size, bias=False)
        self.rotary_emb = DeepSeekRotaryEmbedding(qk_rope_head_dim, max_position_embeddings=max_position_embeddings, base=rope_theta)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Query projection
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # KV projection
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Apply RoPE
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin)

        # Assemble Q and K
        query_states = torch.empty(bsz, self.num_heads, q_len, self.q_head_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        query_states[:, :, :, :self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim:] = q_pe

        key_states = torch.empty(bsz, self.num_heads, q_len, self.q_head_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        key_states[:, :, :, :self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim:] = k_pe

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale

        # Use custom masked softmax kernel
        attention_module.masked_softmax_hip(attn_weights, q_len, self.num_heads, bsz)

        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output


def custom_kernel(inputs):
    hidden_size = 2048
    num_attention_heads = 16
    q_lora_rank = 1536
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    max_position_embeddings = 4096
    
    model = ModelNew(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        max_position_embeddings=max_position_embeddings,
    ).cuda().eval()
    
    with torch.no_grad():
        return model(inputs[0])
