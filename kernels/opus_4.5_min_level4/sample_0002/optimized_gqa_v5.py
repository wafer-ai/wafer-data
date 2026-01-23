import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

os.environ["CXX"] = "hipcc"

from torch.utils.cpp_extension import load_inline

# Fused RoPE + KV expansion kernel
fused_ops_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Simple but efficient RoPE kernel
__global__ void rope_kernel(
    const float* __restrict__ input,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ output,
    const int num_elements,
    const int seq_len,
    const int head_dim,
    const int half_dim
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_elements) return;
    
    // Decode position
    int d = gid % half_dim;
    int remaining = gid / half_dim;
    int s = remaining % seq_len;
    int bh = remaining / seq_len;
    
    int stride = seq_len * head_dim;
    int base_idx = bh * stride + s * head_dim;
    int idx1 = base_idx + d;
    int idx2 = base_idx + d + half_dim;
    
    int cs_idx1 = s * head_dim + d;
    int cs_idx2 = s * head_dim + d + half_dim;
    
    float x1 = input[idx1];
    float x2 = input[idx2];
    float c1 = cos[cs_idx1];
    float s1 = sin[cs_idx1];
    float c2 = cos[cs_idx2];
    float s2 = sin[cs_idx2];
    
    output[idx1] = x1 * c1 - x2 * s1;
    output[idx2] = x2 * c2 + x1 * s2;
}

torch::Tensor rope_forward(torch::Tensor input, torch::Tensor cos, torch::Tensor sin) {
    auto batch_size = input.size(0);
    auto num_heads = input.size(1);
    auto seq_len = input.size(2);
    auto head_dim = input.size(3);
    int half_dim = head_dim / 2;
    
    auto output = torch::empty_like(input);
    
    int num_elements = batch_size * num_heads * seq_len * half_dim;
    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;
    
    rope_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        cos.data_ptr<float>(),
        sin.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements,
        seq_len,
        head_dim,
        half_dim
    );
    
    return output;
}

// Fused KV expansion kernel - expands KV heads without allocating expanded memory
// This creates the expanded tensor directly
__global__ void expand_kv_kernel(
    const float* __restrict__ kv_in,
    float* __restrict__ kv_out,
    const int batch_size,
    const int num_kv_heads,
    const int num_groups,
    const int seq_len,
    const int head_dim
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_kv_heads * num_groups * seq_len * head_dim;
    
    if (gid >= total) return;
    
    // Output layout: [batch, num_kv_heads * num_groups, seq_len, head_dim]
    int d = gid % head_dim;
    int remaining = gid / head_dim;
    int s = remaining % seq_len;
    remaining = remaining / seq_len;
    int expanded_h = remaining % (num_kv_heads * num_groups);
    int b = remaining / (num_kv_heads * num_groups);
    
    // Map expanded head back to original KV head
    int kv_h = expanded_h / num_groups;
    
    // Input index
    int in_idx = b * (num_kv_heads * seq_len * head_dim) + 
                 kv_h * (seq_len * head_dim) + 
                 s * head_dim + d;
    
    kv_out[gid] = kv_in[in_idx];
}

torch::Tensor expand_kv(torch::Tensor kv, int num_groups) {
    auto batch_size = kv.size(0);
    auto num_kv_heads = kv.size(1);
    auto seq_len = kv.size(2);
    auto head_dim = kv.size(3);
    
    auto output = torch::empty({batch_size, num_kv_heads * num_groups, seq_len, head_dim}, 
                               kv.options());
    
    int total = batch_size * num_kv_heads * num_groups * seq_len * head_dim;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    
    expand_kv_kernel<<<num_blocks, block_size>>>(
        kv.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_kv_heads,
        num_groups,
        seq_len,
        head_dim
    );
    
    return output;
}
"""

fused_ops_cpp = """
torch::Tensor rope_forward(torch::Tensor input, torch::Tensor cos, torch::Tensor sin);
torch::Tensor expand_kv(torch::Tensor kv, int num_groups);
"""

fused_ops = load_inline(
    name="fused_ops_v5",
    cpp_sources=fused_ops_cpp,
    cuda_sources=fused_ops_source,
    functions=["rope_forward", "expand_kv"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"],
)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cache = None
        self._sin_cache = None
        self._cache_len = 0

    @torch.no_grad()
    def forward(self, device, seq_len):
        if seq_len != self._cache_len or self._cos_cache is None or self._cos_cache.device != device:
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device))
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cache = emb.cos().unsqueeze(0).unsqueeze(0).contiguous()
            self._sin_cache = emb.sin().unsqueeze(0).unsqueeze(0).contiguous()
            self._cache_len = seq_len
        return self._cos_cache, self._sin_cache


class ModelNew(nn.Module):
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

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(head_dim, max_position_embeddings, rope_theta)
        self.fused_ops = fused_ops

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # Apply RoPE
        cos, sin = self.rotary_emb(hidden_states.device, q_len)
        query_states = self.fused_ops.rope_forward(query_states, cos, sin)
        key_states = self.fused_ops.rope_forward(key_states, cos, sin)

        # Expand KV heads using custom kernel
        key_states = self.fused_ops.expand_kv(key_states, self.num_key_value_groups)
        value_states = self.fused_ops.expand_kv(value_states, self.num_key_value_groups)
        
        # Flash attention
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=None, dropout_p=0.0, is_causal=True, scale=self.softmax_scale
        )

        # Output
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output)


def custom_kernel(inputs):
    hidden_states = inputs[0]
    model = ModelNew(
        hidden_size=4096, num_attention_heads=32, num_key_value_heads=8,
        head_dim=128, max_position_embeddings=4096
    ).cuda().eval()
    with torch.no_grad():
        return model(hidden_states)
