import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused RoPE kernel with vectorized float4 loads
fused_rope_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Float4 vector type for coalesced memory access
typedef struct __align__(16) {
    float x, y, z, w;
} float4;

__device__ __forceinline__ float4 make_float4(float x, float y, float z, float w) {
    float4 f;
    f.x = x; f.y = y; f.z = z; f.w = w;
    return f;
}

// Vectorized RoPE kernel - processes 4 elements at once
__global__ void rope_vectorized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    // Each thread processes a pair of float4 (8 elements total)
    int half_dim = head_dim / 2;
    int vec4_per_half = half_dim / 4;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_vec4 = batch_size * num_heads * seq_len * vec4_per_half;
    
    if (idx < total_vec4) {
        // Calculate position
        int v = idx % vec4_per_half;
        int temp = idx / vec4_per_half;
        int s = temp % seq_len;
        temp = temp / seq_len;
        int h = temp % num_heads;
        int b = temp / num_heads;
        
        // Base offset for this head/seq position
        int base = ((b * num_heads + h) * seq_len + s) * head_dim;
        int d = v * 4;  // Element offset within half dimension
        
        // Load first half (x1) and second half (x2)
        int idx1 = base + d;
        int idx2 = base + d + half_dim;
        
        float4* in_ptr1 = (float4*)(input + idx1);
        float4* in_ptr2 = (float4*)(input + idx2);
        float4 x1 = *in_ptr1;
        float4 x2 = *in_ptr2;
        
        // Load cos/sin for this position
        int cs_base = s * head_dim;
        float4* cos_ptr1 = (float4*)(cos + cs_base + d);
        float4* sin_ptr1 = (float4*)(sin + cs_base + d);
        float4* cos_ptr2 = (float4*)(cos + cs_base + d + half_dim);
        float4* sin_ptr2 = (float4*)(sin + cs_base + d + half_dim);
        
        float4 c1 = *cos_ptr1;
        float4 s1 = *sin_ptr1;
        float4 c2 = *cos_ptr2;
        float4 s2 = *sin_ptr2;
        
        // Apply RoPE: output = input * cos + rotate_half(input) * sin
        // rotate_half maps (x1, x2) -> (-x2, x1)
        float4 out1, out2;
        out1.x = x1.x * c1.x + (-x2.x) * s1.x;
        out1.y = x1.y * c1.y + (-x2.y) * s1.y;
        out1.z = x1.z * c1.z + (-x2.z) * s1.z;
        out1.w = x1.w * c1.w + (-x2.w) * s1.w;
        
        out2.x = x2.x * c2.x + x1.x * s2.x;
        out2.y = x2.y * c2.y + x1.y * s2.y;
        out2.z = x2.z * c2.z + x1.z * s2.z;
        out2.w = x2.w * c2.w + x1.w * s2.w;
        
        // Store results
        float4* out_ptr1 = (float4*)(output + idx1);
        float4* out_ptr2 = (float4*)(output + idx2);
        *out_ptr1 = out1;
        *out_ptr2 = out2;
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
    
    const int block_size = 256;
    int half_dim = head_dim / 2;
    int vec4_per_half = half_dim / 4;
    
    // Flatten cos/sin
    auto cos_flat = cos.contiguous().view({-1});
    auto sin_flat = sin.contiguous().view({-1});
    
    // Process Q
    int total_q_vec4 = batch_size * num_q_heads * seq_len * vec4_per_half;
    int num_blocks_q = (total_q_vec4 + block_size - 1) / block_size;
    
    rope_vectorized_kernel<<<num_blocks_q, block_size>>>(
        q.data_ptr<float>(),
        cos_flat.data_ptr<float>(),
        sin_flat.data_ptr<float>(),
        q_out.data_ptr<float>(),
        batch_size, num_q_heads, seq_len, head_dim
    );
    
    // Process K
    int total_k_vec4 = batch_size * num_kv_heads * seq_len * vec4_per_half;
    int num_blocks_k = (total_k_vec4 + block_size - 1) / block_size;
    
    rope_vectorized_kernel<<<num_blocks_k, block_size>>>(
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
std::vector<torch::Tensor> fused_rope_hip(torch::Tensor q, torch::Tensor k, torch::Tensor cos, torch::Tensor sin);
"""

fused_rope_module = load_inline(
    name="fused_rope_v3",
    cpp_sources=fused_rope_cpp,
    cuda_sources=fused_rope_source,
    functions=["fused_rope_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
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
    Optimized Grouped Query Attention using:
    1. Vectorized fused RoPE kernel with float4 loads
    2. PyTorch's scaled_dot_product_attention for Flash Attention
    3. Memory-efficient KV expansion
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

        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # Apply rotary embeddings with fused kernel
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        query_states, key_states = self.fused_rope.fused_rope_hip(query_states, key_states, cos, sin)

        # Efficient KV expansion - use contiguous for SDPA compatibility
        key_states = key_states[:, :, None, :, :].expand(
            bsz, self.num_kv_heads, self.num_key_value_groups, q_len, self.head_dim
        ).reshape(bsz, self.num_heads, q_len, self.head_dim).contiguous()
        
        value_states = value_states[:, :, None, :, :].expand(
            bsz, self.num_kv_heads, self.num_key_value_groups, q_len, self.head_dim
        ).reshape(bsz, self.num_heads, q_len, self.head_dim).contiguous()

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
