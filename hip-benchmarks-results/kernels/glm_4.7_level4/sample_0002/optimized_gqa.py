import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused GQA attention kernel that avoids explicit KV head expansion
gqa_attention_hip_source = r"""
#include <hip/hip_runtime.h>
#include <math.h>

#define WARP_SIZE 32
#define MAX_SEQ_LEN 2048
#define MAX_HEAD_DIM 128

__global__ void gqa_attention_fused_kernel(
    const float* __restrict__ q,        // [batch, num_heads, seq_len, head_dim]
    const float* __restrict__ k,        // [batch, num_kv_heads, seq_len, head_dim]
    const float* __restrict__ v,        // [batch, num_kv_heads, seq_len, head_dim]
    float* __restrict__ output,         // [batch, num_heads, seq_len, head_dim]
    float softmax_scale,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    int key_value_groups
) {
    // Each block handles one query head and one query position
    const int batch_idx = blockIdx.z;
    const int q_head_idx = blockIdx.y;
    const int q_pos = blockIdx.x;
    
    // Compute corresponding KV head (multiple query heads share same KV head)
    const int kv_head_idx = q_head_idx / key_value_groups;
    
    // Causal mask: only compute for positions <= q_pos
    const int kv_seq_len = min(q_pos + 1, seq_len);
    
    // Lane ID in warp
    const int lane_id = threadIdx.x;
    
    // Check bounds
    if (batch_idx >= batch_size || q_head_idx >= num_heads || q_pos >= seq_len) {
        return;
    }
    
    // Base index for query vector
    const int q_base = ((long long)batch_idx * num_heads + q_head_idx) * seq_len * head_dim + q_pos * head_dim;
    
    // Base index for KV
    const int kv_base = ((long long)batch_idx * num_kv_heads + kv_head_idx) * seq_len * head_dim;
    
    // Shared memory: attention weights + query vector
    __shared__ float w_sh[MAX_SEQ_LEN];
    __shared__ float q_vec_sh[MAX_HEAD_DIM];
    
    // Initialize w_sh to 0
    for (int i = lane_id; i < kv_seq_len; i += WARP_SIZE) {
        w_sh[i] = 0.0f;
    }
    __syncthreads();
    
    // Load query vector into shared memory (all threads cooperatively load)
    for (int i = lane_id; i < head_dim; i += WARP_SIZE) {
        q_vec_sh[i] = q[q_base + i];
    }
    __syncthreads();
    
    // Compute QK^T for each key position
    for (int kv_pos = lane_id; kv_pos < kv_seq_len; kv_pos += WARP_SIZE) {
        int k_base = kv_base + kv_pos * head_dim;
        
        // Compute dot product between query and this key vector
        float sum_qk = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            sum_qk += q_vec_sh[i] * k[k_base + i];
        }
        
        // Add to shared memory
        atomicAdd(&w_sh[kv_pos], sum_qk * softmax_scale);
    }
    __syncthreads();
    
    // Softmax - compute max
    float my_max = -INFINITY;
    for (int i = lane_id; i < kv_seq_len; i += WARP_SIZE) {
        my_max = fmaxf(my_max, w_sh[i]);
    }
    
    // Reduce max across warp
    for (int offset = 16; offset > 0; offset /= 2) {
        my_max = fmaxf(my_max, __shfl_xor(my_max, offset));
    }
    float max_val = __shfl(my_max, 0);
    
    // Compute exp and sum
    float my_exp_sum = 0.0f;
    for (int i = lane_id; i < kv_seq_len; i += WARP_SIZE) {
        float exp_val = expf(w_sh[i] - max_val);
        w_sh[i] = exp_val;
        my_exp_sum += exp_val;
    }
    
    // Reduce sum across warp
    for (int offset = 16; offset > 0; offset /= 2) {
        my_exp_sum += __shfl_xor(my_exp_sum, offset);
    }
    float exp_sum = fmaxf(my_exp_sum, 1e-6f);
    
    // Normalize (each thread handles its own position)
    for (int i = lane_id; i < kv_seq_len; i += WARP_SIZE) {
        w_sh[i] = w_sh[i] / exp_sum;
    }
    
    __syncthreads();
    
    // Compute weighted sum over V (only threads 0..head_dim-1 participate)
    if (lane_id < head_dim) {
        float out_val = 0.0f;
        for (int kv_pos = 0; kv_pos < kv_seq_len; kv_pos++) {
            int v_base = kv_base + kv_pos * head_dim;
            float v_val = v[v_base + lane_id];
            out_val += w_sh[kv_pos] * v_val;
        }
        
        // Write output
        output[q_base + lane_id] = out_val;
    }
}

torch::Tensor gqa_attention_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    float softmax_scale,
    int key_value_groups
) {
    auto batch_size = q.size(0);
    auto num_heads = q.size(1);
    auto num_kv_heads = k.size(1);
    auto seq_len = q.size(2);
    auto head_dim = q.size(3);
    
    auto output = torch::empty_like(q);
    
    const int block_size = 32;  // One warp per block
    dim3 grid(seq_len, num_heads, batch_size);
    
    hipLaunchKernelGGL(
        (gqa_attention_fused_kernel),
        grid, dim3(block_size, 1, 1), 0, 0,
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        output.data_ptr<float>(),
        softmax_scale,
        (int)batch_size,
        (int)num_heads,
        (int)num_kv_heads,
        (int)seq_len,
        (int)head_dim,
        (int)key_value_groups
    );
    
    return output;
}
"""

# Load the fused attention kernel
gqa_attention = load_inline(
    name="gqa_attention",
    cpp_sources=gqa_attention_hip_source,
    functions=["gqa_attention_forward"],
    extra_cflags=["-O3", "-ffast-math"],
    verbose=False,
)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary positional embeddings."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


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
    Grouped Query Attention (GQA) - Optimized with Fused Kernel

    Optimizations:
    1. Replaced standard matmul attention with fused GQA kernel
    2. Avoids explicit KV head expansion (computes implicitly)
    3. Fuses dot product, softmax, and value aggregation in single kernel
    4. Reduces memory traffic by eliminating temporary KV expansion
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

        # Custom fused attention kernel
        self.gqa_kernel = gqa_attention

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
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Compute attention with fused GQA kernel
        # This avoids explicit KV head expansion
        attn_output = self.gqa_kernel.gqa_attention_forward(
            query_states,
            key_states,
            value_states,
            self.softmax_scale,
            self.num_key_value_groups
        )

        # Apply dropout
        if self.attention_dropout > 0.0 and self.training:
            attn_output = F.dropout(attn_output, p=self.attention_dropout, training=True)

        # Reshape and output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output