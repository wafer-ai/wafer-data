import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

os.environ["CXX"] = "hipcc"

from torch.utils.cpp_extension import load_inline

# Fused RoPE kernel with vectorized memory access
fused_ops_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Fused RoPE that processes both Q and K simultaneously
// This allows better memory bandwidth utilization
__global__ void fused_rope_qk_kernel(
    const float* __restrict__ Q_in,
    const float* __restrict__ K_in,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ Q_out,
    float* __restrict__ K_out,
    const int q_batch_heads,    // batch * num_q_heads
    const int k_batch_heads,    // batch * num_kv_heads
    const int batch_size,
    const int num_q_heads,
    const int num_kv_heads,
    const int seq_len,
    const int head_dim,
    const int half_dim
) {
    // Each thread processes one half-dimension pair for either Q or K
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // First process Q, then K
    int q_total = q_batch_heads * seq_len * half_dim;
    int k_total = k_batch_heads * seq_len * half_dim;
    
    if (gid < q_total) {
        // Processing Q
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
        
        float x1 = Q_in[idx1];
        float x2 = Q_in[idx2];
        float c1 = cos[cs_idx1];
        float s1 = sin[cs_idx1];
        float c2 = cos[cs_idx2];
        float s2 = sin[cs_idx2];
        
        Q_out[idx1] = x1 * c1 + (-x2) * s1;
        Q_out[idx2] = x2 * c2 + x1 * s2;
    }
    else if (gid < q_total + k_total) {
        // Processing K
        int k_gid = gid - q_total;
        int d = k_gid % half_dim;
        int remaining = k_gid / half_dim;
        int s = remaining % seq_len;
        int bh = remaining / seq_len;
        
        int stride = seq_len * head_dim;
        int base_idx = bh * stride + s * head_dim;
        int idx1 = base_idx + d;
        int idx2 = base_idx + d + half_dim;
        
        int cs_idx1 = s * head_dim + d;
        int cs_idx2 = s * head_dim + d + half_dim;
        
        float x1 = K_in[idx1];
        float x2 = K_in[idx2];
        float c1 = cos[cs_idx1];
        float s1 = sin[cs_idx1];
        float c2 = cos[cs_idx2];
        float s2 = sin[cs_idx2];
        
        K_out[idx1] = x1 * c1 + (-x2) * s1;
        K_out[idx2] = x2 * c2 + x1 * s2;
    }
}

std::vector<torch::Tensor> fused_rope_qk(
    torch::Tensor Q, 
    torch::Tensor K, 
    torch::Tensor cos, 
    torch::Tensor sin
) {
    auto batch_size = Q.size(0);
    auto num_q_heads = Q.size(1);
    auto num_kv_heads = K.size(1);
    auto seq_len = Q.size(2);
    auto head_dim = Q.size(3);
    int half_dim = head_dim / 2;
    
    auto Q_out = torch::empty_like(Q);
    auto K_out = torch::empty_like(K);
    
    int q_batch_heads = batch_size * num_q_heads;
    int k_batch_heads = batch_size * num_kv_heads;
    
    int total = (q_batch_heads + k_batch_heads) * seq_len * half_dim;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    
    fused_rope_qk_kernel<<<num_blocks, block_size>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        cos.data_ptr<float>(),
        sin.data_ptr<float>(),
        Q_out.data_ptr<float>(),
        K_out.data_ptr<float>(),
        q_batch_heads,
        k_batch_heads,
        batch_size,
        num_q_heads,
        num_kv_heads,
        seq_len,
        head_dim,
        half_dim
    );
    
    return {Q_out, K_out};
}
"""

fused_ops_cpp = """
std::vector<torch::Tensor> fused_rope_qk(torch::Tensor Q, torch::Tensor K, torch::Tensor cos, torch::Tensor sin);
"""

fused_ops = load_inline(
    name="fused_ops_v4",
    cpp_sources=fused_ops_cpp,
    cuda_sources=fused_ops_source,
    functions=["fused_rope_qk"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"],
)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
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

        self.rotary_emb = RotaryEmbedding(
            head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )
        
        self.fused_ops = fused_ops

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention - use contiguous views
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # Get cached cos/sin for RoPE
        cos, sin = self.rotary_emb(hidden_states.device, q_len)
        
        # Apply fused RoPE to Q and K in single kernel
        query_states, key_states = self.fused_ops.fused_rope_qk(query_states, key_states, cos, sin)

        # Expand KV heads using repeat_interleave
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Use flash attention - highly optimized on AMD GPUs
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=self.softmax_scale,
        )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output)


def custom_kernel(inputs):
    hidden_states = inputs[0]
    
    model = ModelNew(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=4096,
    ).cuda().eval()
    
    with torch.no_grad():
        return model(hidden_states)
