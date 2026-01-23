import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

os.environ["CXX"] = "hipcc"

from torch.utils.cpp_extension import load_inline

# Highly optimized RoPE kernel for AMD MI300X
fused_ops_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Optimized RoPE kernel with coalesced memory access
// Each thread processes one (d, d+half_dim) pair
__global__ void rope_kernel_optimized(
    const float* __restrict__ input,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ output,
    const int batch_heads,
    const int seq_len,
    const int head_dim,
    const int half_dim
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_heads * seq_len * half_dim;
    
    if (gid >= total) return;
    
    // Decode position: [batch*heads, seq, half_dim]
    int d = gid % half_dim;
    int temp = gid / half_dim;
    int s = temp % seq_len;
    int bh = temp / seq_len;
    
    // Calculate indices
    int base = bh * seq_len * head_dim + s * head_dim;
    int idx1 = base + d;
    int idx2 = base + d + half_dim;
    
    // cos/sin: [seq_len, head_dim] - flattened
    int cs1 = s * head_dim + d;
    int cs2 = s * head_dim + d + half_dim;
    
    // Load values
    float x1 = input[idx1];
    float x2 = input[idx2];
    float c1 = cos[cs1];
    float s1 = sin[cs1];
    float c2 = cos[cs2];
    float s2 = sin[cs2];
    
    // Apply rotation
    // rotate_half: first half gets -second_half, second half gets first_half
    // y1 = x1 * cos1 + (-x2) * sin1
    // y2 = x2 * cos2 + x1 * sin2
    output[idx1] = x1 * c1 - x2 * s1;
    output[idx2] = x2 * c2 + x1 * s2;
}

torch::Tensor rope_forward(torch::Tensor input, torch::Tensor cos, torch::Tensor sin) {
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    
    int batch_size = input.size(0);
    int num_heads = input.size(1);
    int seq_len = input.size(2);
    int head_dim = input.size(3);
    int half_dim = head_dim / 2;
    
    auto output = torch::empty_like(input);
    
    int batch_heads = batch_size * num_heads;
    int total = batch_heads * seq_len * half_dim;
    
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    
    rope_kernel_optimized<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        cos.data_ptr<float>(),
        sin.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_heads,
        seq_len,
        head_dim,
        half_dim
    );
    
    return output;
}
"""

fused_ops_cpp = """
torch::Tensor rope_forward(torch::Tensor input, torch::Tensor cos, torch::Tensor sin);
"""

fused_ops = load_inline(
    name="fused_ops_v7",
    cpp_sources=fused_ops_cpp,
    cuda_sources=fused_ops_source,
    functions=["rope_forward"],
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

        # QKV projection
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to [batch, heads, seq, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # RoPE
        cos, sin = self.rotary_emb(hidden_states.device, q_len)
        query_states = self.fused_ops.rope_forward(query_states, cos, sin)
        key_states = self.fused_ops.rope_forward(key_states, cos, sin)

        # Expand KV heads - use repeat_interleave (well-optimized on AMD)
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Flash Attention via SDPA
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=self.softmax_scale
        )

        # Reshape and output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output)


def custom_kernel(inputs):
    hidden_states = inputs[0]
    model = ModelNew(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=4096
    ).cuda().eval()
    with torch.no_grad():
        return model(hidden_states)
