import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused RoPE kernel for better memory efficiency
fused_rope_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void fused_rope_kernel(
    const float* __restrict__ q_in,
    const float* __restrict__ k_in,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim
) {
    // Each thread handles one element
    int total_q = batch_size * num_q_heads * seq_len * head_dim;
    int total_k = batch_size * num_kv_heads * seq_len * head_dim;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process Q
    if (idx < total_q) {
        int half_dim = head_dim / 2;
        int d = idx % head_dim;
        int temp = idx / head_dim;
        int s = temp % seq_len;
        temp = temp / seq_len;
        int h = temp % num_q_heads;
        int b = temp / num_q_heads;
        
        // cos/sin shape: [1, 1, seq_len, head_dim]
        int cs_idx = s * head_dim + d;
        float cos_val = cos[cs_idx];
        float sin_val = sin[cs_idx];
        
        float x1, x2;
        if (d < half_dim) {
            // First half: q_embed = q * cos - q_rotated * sin
            // q_rotated for first half comes from second half with negation
            int other_idx = idx + half_dim;
            x1 = q_in[idx];
            x2 = q_in[other_idx];
            q_out[idx] = x1 * cos_val + (-x2) * sin_val;
        } else {
            // Second half: q_embed = q * cos + q_rotated * sin
            int other_idx = idx - half_dim;
            x1 = q_in[idx];
            x2 = q_in[other_idx];
            q_out[idx] = x1 * cos_val + x2 * sin_val;
        }
    }
    
    // Process K (separate pass for different head count)
    if (idx < total_k) {
        int half_dim = head_dim / 2;
        int d = idx % head_dim;
        int temp = idx / head_dim;
        int s = temp % seq_len;
        temp = temp / seq_len;
        int h = temp % num_kv_heads;
        int b = temp / num_kv_heads;
        
        int cs_idx = s * head_dim + d;
        float cos_val = cos[cs_idx];
        float sin_val = sin[cs_idx];
        
        float x1, x2;
        if (d < half_dim) {
            int other_idx = idx + half_dim;
            x1 = k_in[idx];
            x2 = k_in[other_idx];
            k_out[idx] = x1 * cos_val + (-x2) * sin_val;
        } else {
            int other_idx = idx - half_dim;
            x1 = k_in[idx];
            x2 = k_in[other_idx];
            k_out[idx] = x1 * cos_val + x2 * sin_val;
        }
    }
}

std::vector<torch::Tensor> fused_rope_hip(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor cos,
    torch::Tensor sin
) {
    auto batch_size = q.size(0);
    auto num_q_heads = q.size(1);
    auto num_kv_heads = k.size(1);
    auto seq_len = q.size(2);
    auto head_dim = q.size(3);
    
    auto q_out = torch::empty_like(q);
    auto k_out = torch::empty_like(k);
    
    int total_q = batch_size * num_q_heads * seq_len * head_dim;
    int total_k = batch_size * num_kv_heads * seq_len * head_dim;
    int total = std::max(total_q, total_k);
    
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    
    // Flatten cos/sin for easier indexing
    auto cos_flat = cos.contiguous().view({-1});
    auto sin_flat = sin.contiguous().view({-1});
    
    fused_rope_kernel<<<num_blocks, block_size>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        cos_flat.data_ptr<float>(),
        sin_flat.data_ptr<float>(),
        q_out.data_ptr<float>(),
        k_out.data_ptr<float>(),
        batch_size,
        num_q_heads,
        num_kv_heads,
        seq_len,
        head_dim
    );
    
    return {q_out, k_out};
}
"""

fused_rope_cpp = """
std::vector<torch::Tensor> fused_rope_hip(torch::Tensor q, torch::Tensor k, torch::Tensor cos, torch::Tensor sin);
"""

fused_rope_module = load_inline(
    name="fused_rope",
    cpp_sources=fused_rope_cpp,
    cuda_sources=fused_rope_source,
    functions=["fused_rope_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000.0):
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
        return emb.cos().unsqueeze(0).unsqueeze(0), emb.sin().unsqueeze(0).unsqueeze(0)


class ModelNew(nn.Module):
    """
    Optimized Grouped Query Attention using:
    1. Custom fused RoPE kernel 
    2. PyTorch's scaled_dot_product_attention with enable_gqa for efficient GQA
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.attention_dropout = attention_dropout
        self.softmax_scale = head_dim ** (-0.5)

        # Separate projections for Q, K, V
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )
        
        self.fused_rope = fused_rope_module

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Fallback KV expansion"""
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, seq_len, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # Apply rotary embeddings with fused kernel
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        query_states, key_states = self.fused_rope.fused_rope_hip(query_states, key_states, cos, sin)

        # Expand KV heads to match query heads (required for standard attention)
        key_states = self.repeat_kv(key_states, self.num_key_value_groups)
        value_states = self.repeat_kv(value_states, self.num_key_value_groups)

        # Use scaled_dot_product_attention for efficient computation
        # It handles causal masking internally with is_causal=True
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states, 
            value_states,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
            scale=self.softmax_scale
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output


def custom_kernel(inputs):
    hidden_states = inputs[0]
    
    # Use global config
    hidden_size = 4096
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    max_position_embeddings = 4096
    
    model = ModelNew(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings,
    ).to(hidden_states.device)
    
    return model(hidden_states)
