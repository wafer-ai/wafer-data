import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused attention kernel with masked softmax
attention_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void fused_attention_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    float* __restrict__ output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float softmax_scale
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    
    if (b >= batch_size || h >= num_heads) return;
    
    int out_base = ((b * num_heads + h) * seq_len) * head_dim;
    
    // Process each query position
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        int q_base = ((b * num_heads + h) * seq_len + t) * head_dim;
        
        // Compute attention scores for this query position
        float scores[2048];  // Max seq_len
        
        for (int i = 0; i < seq_len; i++) {
            int k_base = ((b * num_heads + h) * seq_len + i) * head_dim;
            
            // Dot product
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += query[q_base + d] * key[k_base + d];
            }
            
            // Apply causal mask (only pay attention to positions <= current)
            scores[i] = (i <= t) ? dot * softmax_scale : -1e10f;
        }
        
        // Compute softmax
        float max_score = -1e10f;
        for (int i = 0; i <= t; i++) {
            max_score = fmaxf(max_score, scores[i]);
        }
        
        float sum = 0.0f;
        for (int i = 0; i <= t; i++) {
            scores[i] = expf(scores[i] - max_score);
            sum += scores[i];
        }
        
        // Normalize
        for (int i = 0; i <= t; i++) {
            scores[i] /= sum;
        }
        
        // Apply attention to value
        for (int d = 0; d < head_dim; d++) {
            float accum = 0.0f;
            for (int i = 0; i <= t; i++) {
                int v_base = ((b * num_heads + h) * seq_len + i) * head_dim;
                accum += scores[i] * value[v_base + d];
            }
            output[out_base + t * head_dim + d] = accum;
        }
    }
}

torch::Tensor fused_attention_hip(
    torch::Tensor query, torch::Tensor key, torch::Tensor value, float softmax_scale
) {
    auto batch_size = query.size(0);
    auto num_heads = query.size(1);
    auto seq_len = query.size(2);
    auto head_dim = query.size(3);
    
    auto output = torch::empty({batch_size, num_heads, seq_len, head_dim}, 
                               query.options());
    
    dim3 grid(batch_size, num_heads);
    int block_size = 256;
    
    fused_attention_kernel<<<grid, block_size>>>(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        softmax_scale
    );
    
    return output;
}
"""

attention = load_inline(
    name="attention",
    cpp_sources=attention_cpp_source,
    functions=["fused_attention_hip"],
    verbose=False,
)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_fixed(q, k, cos, sin):
    """
    Fixed version of apply_rotary_pos_emb that handles broadcasting correctly.
    
    q: (bsz, num_heads, q_len, head_dim)
    k: (bsz, num_heads, k_len, head_dim) or (bsz, 1, k_len, head_dim) for MQA
    cos, sin: (seq_len, head_dim)
    """
    # Reshape cos/sin to (1, 1, seq_len, head_dim) for proper broadcasting
    cos = cos.view(1, 1, -1, cos.shape[-1])
    sin = sin.view(1, 1, -1, sin.shape[-1])
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


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
    """
    Optimized DeepSeek-V3 Multi-head Latent Attention (MLA)
    
    Key optimizations:
    1. Fixed RoPE application with correct broadcasting
    2. Fused attention kernel with masked softmax
    """

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

        # Query projection with LoRA compression
        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_a_layernorm = DeepSeekRMSNorm(q_lora_rank)
        self.q_b_proj = nn.Linear(q_lora_rank, num_attention_heads * self.q_head_dim, bias=False)

        # KV projection with LoRA compression (MQA-style: shared across heads initially)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = DeepSeekRMSNorm(kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            num_attention_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )

        # Output projection
        self.o_proj = nn.Linear(num_attention_heads * v_head_dim, hidden_size, bias=False)

        # Rotary embeddings
        self.rotary_emb = DeepSeekRotaryEmbedding(
            qk_rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )
        
        # Load fused attention kernel
        self.fused_attention_hip = attention.fused_attention_hip

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Query projection with LoRA compression
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)

        # Split query into nope (non-positional) and rope (positional) components
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # KV projection with compression
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        # Expand compressed KV
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        kv = kv.transpose(1, 2)

        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Apply rotary embeddings to positional components only (FIXED version)
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        q_pe, k_pe = apply_rotary_pos_emb_fixed(q_pe, k_pe, cos, sin)

        # Assemble full query and key states
        query_states = torch.empty(bsz, self.num_heads, q_len, self.q_head_dim,
                                   device=hidden_states.device, dtype=hidden_states.dtype)
        query_states[:, :, :, :self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim:] = q_pe

        key_states = torch.empty(bsz, self.num_heads, q_len, self.q_head_dim,
                                 device=hidden_states.device, dtype=hidden_states.dtype)
        key_states[:, :, :, :self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim:] = k_pe

        # Compute attention using fused kernel
        attn_output = self.fused_attention_hip(query_states, key_states, value_states, self.softmax_scale)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output