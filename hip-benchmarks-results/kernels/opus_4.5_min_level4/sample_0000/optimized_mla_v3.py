import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused RMSNorm kernel with vectorized loads
rmsnorm_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

template<int BLOCK_SIZE>
__global__ void rmsnorm_kernel_opt(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int hidden_size,
    float eps,
    int total_tokens
) {
    int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;
    
    const float* token_input = input + token_idx * hidden_size;
    float* token_output = output + token_idx * hidden_size;
    
    float sum_sq = 0.0f;
    
    int vec_size = hidden_size / 4;
    
    const float4* input_vec = reinterpret_cast<const float4*>(token_input);
    for (int i = threadIdx.x; i < vec_size; i += BLOCK_SIZE) {
        float4 val = input_vec[i];
        sum_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }
    
    // Warp reduction
    #pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        sum_sq += __shfl_xor(sum_sq, offset);
    }
    
    // Cross-warp reduction
    __shared__ float shared_sum[8];
    int warp_id = threadIdx.x / 64;
    int lane = threadIdx.x % 64;
    
    if (lane == 0) {
        shared_sum[warp_id] = sum_sq;
    }
    __syncthreads();
    
    if (threadIdx.x < 8) {
        sum_sq = shared_sum[threadIdx.x];
        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            sum_sq += __shfl_xor(sum_sq, offset);
        }
    }
    
    __shared__ float rsqrt_var;
    if (threadIdx.x == 0) {
        rsqrt_var = rsqrtf(sum_sq / hidden_size + eps);
    }
    __syncthreads();
    
    float4* output_vec = reinterpret_cast<float4*>(token_output);
    const float4* weight_vec = reinterpret_cast<const float4*>(weight);
    
    for (int i = threadIdx.x; i < vec_size; i += BLOCK_SIZE) {
        float4 val = input_vec[i];
        float4 w = weight_vec[i];
        float4 out;
        out.x = val.x * rsqrt_var * w.x;
        out.y = val.y * rsqrt_var * w.y;
        out.z = val.z * rsqrt_var * w.z;
        out.w = val.w * rsqrt_var * w.w;
        output_vec[i] = out;
    }
}

torch::Tensor rmsnorm_hip(torch::Tensor input, torch::Tensor weight, float eps) {
    auto output = torch::empty_like(input);
    
    int hidden_size = input.size(-1);
    int total_tokens = input.numel() / hidden_size;
    
    rmsnorm_kernel_opt<256><<<total_tokens, 256>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        hidden_size,
        eps,
        total_tokens
    );
    
    return output;
}
"""

# Compile RMSNorm module
rmsnorm_module = load_inline(
    name="rmsnorm_hip_v3",
    cpp_sources="torch::Tensor rmsnorm_hip(torch::Tensor input, torch::Tensor weight, float eps);",
    cuda_sources=rmsnorm_source,
    functions=["rmsnorm_hip"],
    verbose=False,
    extra_cuda_cflags=["-O3"]
)


class DeepSeekRMSNormOptimized(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return rmsnorm_module.rmsnorm_hip(hidden_states.contiguous(), self.weight, self.variance_epsilon)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
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
    """
    Optimized DeepSeek-V3 Multi-head Latent Attention (MLA)
    Uses scaled_dot_product_attention with is_causal=True for efficiency
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
        self.q_a_layernorm = DeepSeekRMSNormOptimized(q_lora_rank)
        self.q_b_proj = nn.Linear(q_lora_rank, num_attention_heads * self.q_head_dim, bias=False)

        # KV projection with LoRA compression
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = DeepSeekRMSNormOptimized(kv_lora_rank)
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Query projection with LoRA compression
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)

        # Split query into nope and rope components
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

        # Apply rotary embeddings to positional components only
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin)

        # Assemble full query and key states efficiently using torch.cat
        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat([k_nope, k_pe.expand(-1, self.num_heads, -1, -1)], dim=-1)

        # Compute attention - use manual implementation for correctness
        # (F.scaled_dot_product_attention with is_causal=True doesn't match exactly)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(q_len, q_len, device=hidden_states.device, dtype=torch.bool),
            diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output


def custom_kernel(inputs):
    # Initialize model configuration
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
