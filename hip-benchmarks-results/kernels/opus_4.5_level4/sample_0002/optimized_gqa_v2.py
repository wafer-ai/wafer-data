import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused RoPE kernel with vectorized loads
fused_rope_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Process Q and K with separate kernels for better occupancy
__global__ void rope_q_kernel(
    const float* __restrict__ q_in,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ q_out,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int half_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * seq_len * half_dim;
    
    if (idx < total) {
        // Calculate indices for first half dimension
        int d = idx % half_dim;
        int temp = idx / half_dim;
        int s = temp % seq_len;
        temp = temp / seq_len;
        int h = temp % num_heads;
        int b = temp / num_heads;
        
        // Base index into q tensor
        int base_idx = ((b * num_heads + h) * seq_len + s) * head_dim;
        int idx1 = base_idx + d;
        int idx2 = base_idx + d + half_dim;
        
        // cos/sin indices
        int cs_idx1 = s * head_dim + d;
        int cs_idx2 = s * head_dim + d + half_dim;
        
        float cos1 = cos[cs_idx1];
        float sin1 = sin[cs_idx1];
        float cos2 = cos[cs_idx2];
        float sin2 = sin[cs_idx2];
        
        float x1 = q_in[idx1];
        float x2 = q_in[idx2];
        
        // rotate_half: (-x2, x1)
        // q_embed = q * cos + rotate_half(q) * sin
        q_out[idx1] = x1 * cos1 + (-x2) * sin1;
        q_out[idx2] = x2 * cos2 + x1 * sin2;
    }
}

__global__ void rope_k_kernel(
    const float* __restrict__ k_in,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ k_out,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int half_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * seq_len * half_dim;
    
    if (idx < total) {
        int d = idx % half_dim;
        int temp = idx / half_dim;
        int s = temp % seq_len;
        temp = temp / seq_len;
        int h = temp % num_heads;
        int b = temp / num_heads;
        
        int base_idx = ((b * num_heads + h) * seq_len + s) * head_dim;
        int idx1 = base_idx + d;
        int idx2 = base_idx + d + half_dim;
        
        int cs_idx1 = s * head_dim + d;
        int cs_idx2 = s * head_dim + d + half_dim;
        
        float cos1 = cos[cs_idx1];
        float sin1 = sin[cs_idx1];
        float cos2 = cos[cs_idx2];
        float sin2 = sin[cs_idx2];
        
        float x1 = k_in[idx1];
        float x2 = k_in[idx2];
        
        k_out[idx1] = x1 * cos1 + (-x2) * sin1;
        k_out[idx2] = x2 * cos2 + x1 * sin2;
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
    auto half_dim = head_dim / 2;
    
    auto q_out = torch::empty_like(q);
    auto k_out = torch::empty_like(k);
    
    const int block_size = 256;
    
    // Launch Q kernel
    int total_q = batch_size * num_q_heads * seq_len * half_dim;
    int num_blocks_q = (total_q + block_size - 1) / block_size;
    
    auto cos_flat = cos.contiguous().view({-1});
    auto sin_flat = sin.contiguous().view({-1});
    
    rope_q_kernel<<<num_blocks_q, block_size>>>(
        q.data_ptr<float>(),
        cos_flat.data_ptr<float>(),
        sin_flat.data_ptr<float>(),
        q_out.data_ptr<float>(),
        batch_size, num_q_heads, seq_len, head_dim, half_dim
    );
    
    // Launch K kernel
    int total_k = batch_size * num_kv_heads * seq_len * half_dim;
    int num_blocks_k = (total_k + block_size - 1) / block_size;
    
    rope_k_kernel<<<num_blocks_k, block_size>>>(
        k.data_ptr<float>(),
        cos_flat.data_ptr<float>(),
        sin_flat.data_ptr<float>(),
        k_out.data_ptr<float>(),
        batch_size, num_kv_heads, seq_len, head_dim, half_dim
    );
    
    return {q_out, k_out};
}
"""

fused_rope_cpp = """
std::vector<torch::Tensor> fused_rope_hip(torch::Tensor q, torch::Tensor k, torch::Tensor cos, torch::Tensor sin);
"""

fused_rope_module = load_inline(
    name="fused_rope_v2",
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
        
        # Pre-compute cos/sin for common sequence lengths
        self._cos_cached = None
        self._sin_cached = None
        self._cached_seq_len = 0

    @torch.no_grad()
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        
        # Use cached values if possible
        if seq_len <= self._cached_seq_len and self._cos_cached is not None:
            return self._cos_cached[:, :, :seq_len, :], self._sin_cached[:, :, :seq_len, :]
        
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(x.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        
        # Cache for reuse
        self._cos_cached = cos
        self._sin_cached = sin
        self._cached_seq_len = seq_len
        
        return cos, sin


class ModelNew(nn.Module):
    """
    Optimized Grouped Query Attention using:
    1. Custom fused RoPE kernel 
    2. PyTorch's scaled_dot_product_attention for efficient Flash Attention
    3. Efficient KV expansion using expand instead of repeat where possible
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Project Q, K, V - these are the most expensive ops
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

        # Efficient KV expansion using expand (no memory copy when possible)
        # Shape: [bsz, num_kv_heads, seq_len, head_dim] -> [bsz, num_heads, seq_len, head_dim]
        key_states = key_states[:, :, None, :, :].expand(
            bsz, self.num_kv_heads, self.num_key_value_groups, q_len, self.head_dim
        ).reshape(bsz, self.num_heads, q_len, self.head_dim)
        
        value_states = value_states[:, :, None, :, :].expand(
            bsz, self.num_kv_heads, self.num_key_value_groups, q_len, self.head_dim
        ).reshape(bsz, self.num_heads, q_len, self.head_dim)

        # Use scaled_dot_product_attention for efficient computation
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
