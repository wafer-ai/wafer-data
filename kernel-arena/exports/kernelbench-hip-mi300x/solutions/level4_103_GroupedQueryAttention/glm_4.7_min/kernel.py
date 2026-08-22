import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized kernel for efficient KV head expansion
repeat_kv_source = """
#include <hip/hip_runtime.h>

// More efficient KV head expansion kernel
__global__ void repeat_kv_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int num_kv_heads,
    const int n_rep,
    const int seq_len,
    const int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_kv_heads * n_rep * seq_len * head_dim;
    
    if (idx >= total_elements) return;
    
    // Decompose linear index
    int dim = idx % head_dim;
    idx /= head_dim;
    int pos = idx % seq_len;
    idx /= seq_len;
    int out_head = idx % (num_kv_heads * n_rep);
    int batch = idx / (num_kv_heads * n_rep);
    
    // Map output head input head
    int kv_head = out_head / n_rep;
    
    int in_idx = batch * num_kv_heads * seq_len * head_dim + kv_head * seq_len * head_dim + pos * head_dim + dim;
    
    output[idx] = input[in_idx];
}

torch::Tensor repeat_kv_hip(
    torch::Tensor hidden_states,
    int n_rep
) {
    auto batch = hidden_states.size(0);
    auto num_kv_heads = hidden_states.size(1);
    auto seq_len = hidden_states.size(2);
    auto head_dim = hidden_states.size(3);
    
    if (n_rep == 1) {
        return hidden_states;
    }
    
    auto output = torch::zeros({batch, num_kv_heads * n_rep, seq_len, head_dim}, hidden_states.options());
    
    int total_elements = batch * num_kv_heads * n_rep * seq_len * head_dim;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    repeat_kv_kernel<<<num_blocks, block_size>>>(
        hidden_states.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        num_kv_heads,
        n_rep,
        seq_len,
        head_dim
    );
    
    return output;
}
"""

repeat_kv_lib = load_inline(
    name="repeat_kv_lib",
    cpp_sources=repeat_kv_source,
    functions=["repeat_kv_hip"],
    verbose=True,
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
    Optimized Grouped Query Attention (GQA).
    
    Key optimizations:
    1. Custom HIP kernel for efficient KV head expansion
    2. Use F.scaled_dot_product_attention for highly optimized attention computation
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
        
        # Repeat KV library
        self.repeat_kv_lib = repeat_kv_lib

    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin):
        """Apply rotary positional embeddings."""
        q_embed = (q * cos) + (ModelNew.rotate_half(q) * sin)
        k_embed = (k * cos) + (ModelNew.rotate_half(k) * sin)
        return q_embed, k_embed
    apply_rotary_pos_emb = staticmethod(apply_rotary_pos_emb)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # OPTIMIZED: Efficient KV expansion using custom HIP kernel
        key_states = self.repeat_kv_lib.repeat_kv_hip(key_states, self.num_key_value_groups)
        value_states = self.repeat_kv_lib.repeat_kv_hip(value_states, self.num_key_value_groups)

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(q_len, q_len, device=hidden_states.device, dtype=torch.bool),
            diagonal=1
        )

        # OPTIMIZED: Use scaled_dot_product_attention for highly optimized computation
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.softmax_scale
        )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output