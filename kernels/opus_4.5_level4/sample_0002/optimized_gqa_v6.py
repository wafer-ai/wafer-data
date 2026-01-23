import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused RoPE kernel  
fused_rope_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void rope_kernel(
    const float* __restrict__ input,
    const float* __restrict__ cos_data,
    const float* __restrict__ sin_data,
    float* __restrict__ output,
    int num_elements,
    int num_heads,
    int seq_len,
    int head_dim
) {
    int half_dim = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements) {
        int d_pair = idx % half_dim;
        int temp = idx / half_dim;
        int s = temp % seq_len;
        temp = temp / seq_len;
        int h = temp % num_heads;
        int b = temp / num_heads;
        
        int base = ((b * num_heads + h) * seq_len + s) * head_dim;
        int idx1 = base + d_pair;
        int idx2 = base + d_pair + half_dim;
        
        int cs_idx1 = s * head_dim + d_pair;
        int cs_idx2 = s * head_dim + d_pair + half_dim;
        
        float c1 = cos_data[cs_idx1];
        float s1 = sin_data[cs_idx1];
        float c2 = cos_data[cs_idx2];
        float s2 = sin_data[cs_idx2];
        
        float x1 = input[idx1];
        float x2 = input[idx2];
        
        // RoPE: first half uses -x2, second half uses +x1
        output[idx1] = x1 * c1 - x2 * s1;
        output[idx2] = x2 * c2 + x1 * s2;
    }
}

torch::Tensor apply_rope_hip(
    torch::Tensor input,
    torch::Tensor cos_tensor,
    torch::Tensor sin_tensor
) {
    auto batch_size = input.size(0);
    auto num_heads = input.size(1);
    auto seq_len = input.size(2);
    auto head_dim = input.size(3);
    
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    int half_dim = head_dim / 2;
    int total = batch_size * num_heads * seq_len * half_dim;
    int num_blocks = (total + block_size - 1) / block_size;
    
    auto cos_flat = cos_tensor.contiguous().view({-1});
    auto sin_flat = sin_tensor.contiguous().view({-1});
    
    rope_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        cos_flat.data_ptr<float>(),
        sin_flat.data_ptr<float>(),
        output.data_ptr<float>(),
        total, num_heads, seq_len, head_dim
    );
    
    return output;
}
"""

fused_rope_cpp = """
torch::Tensor apply_rope_hip(torch::Tensor input, torch::Tensor cos_tensor, torch::Tensor sin_tensor);
"""

fused_rope_module = load_inline(
    name="fused_rope_v6",
    cpp_sources=fused_rope_cpp,
    cuda_sources=fused_rope_source,
    functions=["apply_rope_hip"],
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
        freqs = torch.outer(t, self.inv_freq.to(x.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(0).unsqueeze(0).contiguous(), emb.sin().unsqueeze(0).unsqueeze(0).contiguous()


class ModelNew(nn.Module):
    """
    Optimized GQA with fused QKV projection
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

        # Fused QKV projection for better memory efficiency
        q_size = num_attention_heads * head_dim
        kv_size = num_key_value_heads * head_dim
        self.qkv_proj = nn.Linear(hidden_size, q_size + 2 * kv_size, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)
        
        # Store sizes for splitting
        self.q_size = q_size
        self.kv_size = kv_size

        self.rotary_emb = RotaryEmbedding(
            head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )
        
        self.rope_module = fused_rope_module

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Fused QKV projection - single matmul instead of three
        qkv = self.qkv_proj(hidden_states)
        
        # Split into Q, K, V
        query_states = qkv[..., :self.q_size]
        key_states = qkv[..., self.q_size:self.q_size + self.kv_size]
        value_states = qkv[..., self.q_size + self.kv_size:]

        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings using custom kernel
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        query_states = self.rope_module.apply_rope_hip(query_states, cos, sin)
        key_states = self.rope_module.apply_rope_hip(key_states, cos, sin)

        # Efficient KV expansion
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Flash Attention
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
