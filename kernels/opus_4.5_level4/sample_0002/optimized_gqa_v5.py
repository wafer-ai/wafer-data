import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized fused RoPE kernel 
fused_rope_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Single kernel for both Q and K with warp-level optimization
__global__ void rope_fused_qk_kernel(
    const float* __restrict__ q_in,
    const float* __restrict__ k_in,
    const float* __restrict__ cos_data,
    const float* __restrict__ sin_data,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    int total_q,
    int total_k
) {
    int half_dim = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process Q
    if (idx < total_q) {
        int d_pair = idx % half_dim;
        int temp = idx / half_dim;
        int s = temp % seq_len;
        temp = temp / seq_len;
        int h = temp % num_q_heads;
        int b = temp / num_q_heads;
        
        int base = ((b * num_q_heads + h) * seq_len + s) * head_dim;
        int idx1 = base + d_pair;
        int idx2 = base + d_pair + half_dim;
        
        int cs_idx1 = s * head_dim + d_pair;
        int cs_idx2 = s * head_dim + d_pair + half_dim;
        
        float c1 = cos_data[cs_idx1];
        float s1 = sin_data[cs_idx1];
        float c2 = cos_data[cs_idx2];
        float s2 = sin_data[cs_idx2];
        
        float x1 = q_in[idx1];
        float x2 = q_in[idx2];
        
        q_out[idx1] = x1 * c1 - x2 * s1;
        q_out[idx2] = x2 * c2 + x1 * s2;
    }
    
    // Process K (reuse the same thread if idx < total_k)
    if (idx < total_k) {
        int d_pair = idx % half_dim;
        int temp = idx / half_dim;
        int s = temp % seq_len;
        temp = temp / seq_len;
        int h = temp % num_kv_heads;
        int b = temp / num_kv_heads;
        
        int base = ((b * num_kv_heads + h) * seq_len + s) * head_dim;
        int idx1 = base + d_pair;
        int idx2 = base + d_pair + half_dim;
        
        int cs_idx1 = s * head_dim + d_pair;
        int cs_idx2 = s * head_dim + d_pair + half_dim;
        
        float c1 = cos_data[cs_idx1];
        float s1 = sin_data[cs_idx1];
        float c2 = cos_data[cs_idx2];
        float s2 = sin_data[cs_idx2];
        
        float x1 = k_in[idx1];
        float x2 = k_in[idx2];
        
        k_out[idx1] = x1 * c1 - x2 * s1;
        k_out[idx2] = x2 * c2 + x1 * s2;
    }
}

std::vector<torch::Tensor> fused_rope_hip(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor cos_tensor,
    torch::Tensor sin_tensor
) {
    auto batch_size = q.size(0);
    auto num_q_heads = q.size(1);
    auto num_kv_heads = k.size(1);
    auto seq_len = q.size(2);
    auto head_dim = q.size(3);
    
    auto q_out = torch::empty_like(q);
    auto k_out = torch::empty_like(k);
    
    const int block_size = 256;
    int half_dim = head_dim / 2;
    
    auto cos_flat = cos_tensor.contiguous().view({-1});
    auto sin_flat = sin_tensor.contiguous().view({-1});
    
    int total_q = batch_size * num_q_heads * seq_len * half_dim;
    int total_k = batch_size * num_kv_heads * seq_len * half_dim;
    int total = std::max(total_q, total_k);
    int num_blocks = (total + block_size - 1) / block_size;
    
    rope_fused_qk_kernel<<<num_blocks, block_size>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        cos_flat.data_ptr<float>(),
        sin_flat.data_ptr<float>(),
        q_out.data_ptr<float>(),
        k_out.data_ptr<float>(),
        batch_size, num_q_heads, num_kv_heads, seq_len, head_dim,
        total_q, total_k
    );
    
    return {q_out, k_out};
}
"""

fused_rope_cpp = """
std::vector<torch::Tensor> fused_rope_hip(torch::Tensor q, torch::Tensor k, torch::Tensor cos_tensor, torch::Tensor sin_tensor);
"""

fused_rope_module = load_inline(
    name="fused_rope_v5",
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
        
        self._cos_cached = None
        self._sin_cached = None
        self._cached_seq_len = 0

    @torch.no_grad()
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        
        if seq_len <= self._cached_seq_len and self._cos_cached is not None and self._cos_cached.device == x.device:
            return self._cos_cached[:, :, :seq_len, :], self._sin_cached[:, :, :seq_len, :]
        
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(x.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0).contiguous()
        sin = emb.sin().unsqueeze(0).unsqueeze(0).contiguous()
        
        self._cos_cached = cos
        self._sin_cached = sin
        self._cached_seq_len = seq_len
        
        return cos, sin


class ModelNew(nn.Module):
    """
    Optimized Grouped Query Attention
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

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )
        
        self.fused_rope = fused_rope_module

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention - using reshape instead of view+transpose for efficiency
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        query_states, key_states = self.fused_rope.fused_rope_hip(query_states, key_states, cos, sin)

        # Efficient KV expansion with minimal memory operations
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Use scaled_dot_product_attention (Flash Attention)
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
        attn_output = self.o_proj(attn_output)

        return attn_output


def custom_kernel(inputs):
    hidden_states = inputs[0]
    
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
