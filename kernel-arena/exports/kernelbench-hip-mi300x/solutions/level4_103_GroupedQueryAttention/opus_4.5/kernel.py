import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized RoPE kernel with float2 for better memory coalescing
fused_rope_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void rope_kernel_float2(
    const float* __restrict__ input,
    const float* __restrict__ cos_data,
    const float* __restrict__ sin_data,
    float* __restrict__ output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    int half_dim = head_dim / 2;
    int float2_per_half = half_dim / 2;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * seq_len * float2_per_half;
    
    if (idx < total) {
        int v = idx % float2_per_half;
        int temp = idx / float2_per_half;
        int s = temp % seq_len;
        temp = temp / seq_len;
        int h = temp % num_heads;
        int b = temp / num_heads;
        
        int base = ((b * num_heads + h) * seq_len + s) * head_dim;
        int d = v * 2;
        
        int idx1 = base + d;
        int idx2 = base + d + half_dim;
        
        // Load using float2 for coalesced access
        float2 x1 = *reinterpret_cast<const float2*>(input + idx1);
        float2 x2 = *reinterpret_cast<const float2*>(input + idx2);
        
        int cs_base = s * head_dim;
        float2 c1 = *reinterpret_cast<const float2*>(cos_data + cs_base + d);
        float2 s1 = *reinterpret_cast<const float2*>(sin_data + cs_base + d);
        float2 c2 = *reinterpret_cast<const float2*>(cos_data + cs_base + d + half_dim);
        float2 s2 = *reinterpret_cast<const float2*>(sin_data + cs_base + d + half_dim);
        
        float2 out1, out2;
        out1.x = x1.x * c1.x - x2.x * s1.x;
        out1.y = x1.y * c1.y - x2.y * s1.y;
        out2.x = x2.x * c2.x + x1.x * s2.x;
        out2.y = x2.y * c2.y + x1.y * s2.y;
        
        *reinterpret_cast<float2*>(output + idx1) = out1;
        *reinterpret_cast<float2*>(output + idx2) = out2;
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
    int float2_per_half = half_dim / 2;
    
    auto cos_flat = cos_tensor.contiguous().view({-1});
    auto sin_flat = sin_tensor.contiguous().view({-1});
    
    // Process Q
    int total_q = batch_size * num_q_heads * seq_len * float2_per_half;
    int num_blocks_q = (total_q + block_size - 1) / block_size;
    
    rope_kernel_float2<<<num_blocks_q, block_size>>>(
        q.data_ptr<float>(),
        cos_flat.data_ptr<float>(),
        sin_flat.data_ptr<float>(),
        q_out.data_ptr<float>(),
        batch_size, num_q_heads, seq_len, head_dim
    );
    
    // Process K
    int total_k = batch_size * num_kv_heads * seq_len * float2_per_half;
    int num_blocks_k = (total_k + block_size - 1) / block_size;
    
    rope_kernel_float2<<<num_blocks_k, block_size>>>(
        k.data_ptr<float>(),
        cos_flat.data_ptr<float>(),
        sin_flat.data_ptr<float>(),
        k_out.data_ptr<float>(),
        batch_size, num_kv_heads, seq_len, head_dim
    );
    
    return {q_out, k_out};
}
"""

fused_rope_cpp = """
std::vector<torch::Tensor> fused_rope_hip(torch::Tensor q, torch::Tensor k, torch::Tensor cos_tensor, torch::Tensor sin_tensor);
"""

fused_rope_module = load_inline(
    name="fused_rope_v7",
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
        freqs = torch.outer(t, self.inv_freq.to(x.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(0).unsqueeze(0).contiguous(), emb.sin().unsqueeze(0).unsqueeze(0).contiguous()


class ModelNew(nn.Module):
    """
    Optimized GQA - uses native GQA in SDPA when available
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
        
        self.rope_module = fused_rope_module

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        query_states, key_states = self.rope_module.fused_rope_hip(query_states, key_states, cos, sin)

        # Try to use native GQA support in SDPA
        try:
            # PyTorch 2.4+ has enable_gqa parameter
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states, 
                value_states,
                attn_mask=None,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=True,
                scale=self.softmax_scale,
                enable_gqa=True  # Native GQA support - avoids KV expansion
            )
        except TypeError:
            # Fallback: expand KV heads manually
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
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
