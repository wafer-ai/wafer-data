import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Simple and correct GQA attention kernel
gqa_attention_cpp_source = """
#include <hip/hip_runtime.h>
#include <math.h>

#define MAX_HEAD_DIM 128
#define NEG_INF (-1e20f)

// Helper function to compute rotary embedding
__device__ __forceinline__ void apply_rope(float& x, float& y, float cos_val, float sin_val) {
    float x_new = x * cos_val - y * sin_val;
    float y_new = x * sin_val + y * cos_val;
    x = x_new;
    y = y_new;
}

__global__ void gqa_attention_kernel(
    const float* __restrict__ q,      // [batch, num_heads, seq_len, head_dim]
    const float* __restrict__ k,      // [batch, num_kv_heads, seq_len, head_dim]
    const float* __restrict__ v,      // [batch, num_kv_heads, seq_len, head_dim]
    float* __restrict__ out,          // [batch, num_heads, seq_len, head_dim]
    const float* __restrict__ cos,    // [1, 1, seq_len, head_dim]
    const float* __restrict__ sin,    // [1, 1, seq_len, head_dim]
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    float softmax_scale
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int q_pos = blockIdx.z;
    
    // Each block processes one query position
    int group_size = num_heads / num_kv_heads;
    int kv_head_idx = head_idx / group_size;
    
    int tid = threadIdx.x;
    
    // Load query vector in shared memory (cooperative)
    __shared__ float shared_q[MAX_HEAD_DIM];
    
    for (int d = tid; d < head_dim; d += blockDim.x) {
        int q_idx = ((batch_idx * num_heads + head_idx) * seq_len + q_pos) * head_dim + d;
        shared_q[d] = q[q_idx];
    }
    __syncthreads();
    
    // Apply RoPE to query (in pairs)
    if (tid < head_dim / 2) {
        int d = tid * 2;
        int cos_sin_idx = q_pos * head_dim + d;
        apply_rope(shared_q[d], shared_q[d + 1], cos[cos_sin_idx], sin[cos_sin_idx]);
    }
    __syncthreads();
    
    // Shared memory for scores
    __shared__ float scores[2048];  // Max seq_len = 2048
    
    // Compute scores for all keys (0 to q_pos)
    for (int k_pos = tid; k_pos <= q_pos; k_pos += blockDim.x) {
        float score = 0.0f;
        
        // Compute dot product between q and k
        for (int d = 0; d < head_dim; d += 2) {
            int k_idx = ((batch_idx * num_kv_heads + kv_head_idx) * seq_len + k_pos) * head_dim + d;
            float k_val = k[k_idx];
            float k_val2 = k[k_idx + 1];
            
            if (d + 1 < head_dim) {
                // Apply RoPE to key pair
                int cos_sin_idx = k_pos * head_dim + d;
                apply_rope(k_val, k_val2, cos[cos_sin_idx], sin[cos_sin_idx]);
                
                // Accumulate dot product with RoPE-applied query
                score += shared_q[d] * k_val + shared_q[d + 1] * k_val2;
            } else {
                score += shared_q[d] * k_val;
            }
        }
        
        // Apply softmax scaling
        scores[k_pos] = score * softmax_scale;
    }
    
    __syncthreads();
    
    // Compute softmax (numerically stable) - find max
    __shared__ float max_score;
    if (tid == 0) max_score = NEG_INF;
    __syncthreads();
    
    // Find max in first warp
    if (tid < WARP_SIZE) {
        float local_max = NEG_INF;
        for (int k_pos = tid; k_pos <= q_pos; k_pos += WARP_SIZE) {
            local_max = fmaxf(local_max, scores[k_pos]);
        }
        
        // Warp reduction
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            local_max = fmaxf(local_max, __shfl_down(local_max, offset));
        }
        
        // Write warp max
        if (tid == 0) {
            max_score = local_max;
        }
    }
    __syncthreads();
    
    // Compute exp scores
    for (int k_pos = tid; k_pos <= q_pos; k_pos += blockDim.x) {
        scores[k_pos] = expf(scores[k_pos] - max_score);
    }
    __syncthreads();
    
    // Compute sum of exponentials
    __shared__ float sum_exp;
    if (tid == 0) sum_exp = 0.0f;
    __syncthreads();
    
    // Sum in first warp    
    if (tid < WARP_SIZE) {
        float local_sum = 0.0f;
        for (int k_pos = tid; k_pos <= q_pos; k_pos += WARP_SIZE) {
            local_sum += scores[k_pos];
        }
        
        // Warp reduction
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            local_sum += __shfl_down(local_sum, offset);
        }
        
        // Write result
        if (tid == 0) {
            sum_exp = local_sum;
        }
    }
    __syncthreads();
    
    // Now compute output by accumulating weighted values
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float out_val = 0.0f;
        
        for (int k_pos = 0; k_pos <= q_pos; k_pos++) {
            float weight = scores[k_pos] / fmaxf(sum_exp, 1e-30f);
            int v_idx = ((batch_idx * num_kv_heads + kv_head_idx) * seq_len + k_pos) * head_dim + d;
            out_val += v[v_idx] * weight;
        }
        
        // Store output
        int out_idx = ((batch_idx * num_heads + head_idx) * seq_len + q_pos) * head_dim + d;
        out[out_idx] = out_val;
    }
}

torch::Tensor gqa_attention_hip(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor cos, torch::Tensor sin,
    int num_kv_heads, float softmax_scale
) {
    int batch_size = q.size(0);
    int num_heads = q.size(1);
    int seq_len = q.size(2);
    int head_dim = q.size(3);
    
    auto out = torch::zeros_like(q);
    
    // Simple launch: one block per (batch, head, query position)
    dim3 grid(batch_size, num_heads, seq_len);
    dim3 block(256);  // Threads per block
    
    gqa_attention_kernel<<<grid, block, 0>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        cos.data_ptr<float>(),
        sin.data_ptr<float>(),
        batch_size,
        num_heads,
        num_kv_heads,
        seq_len,
        head_dim,
        softmax_scale
    );
    
    return out;
}
"""

gqa_attention = load_inline(
    name="gqa_attention",
    cpp_sources=gqa_attention_cpp_source,
    functions=["gqa_attention_hip"],
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
        
        # Fused GQA attention kernel
        self.gqa_attention = gqa_attention

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

        # Get rotary embeddings
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)

        # Fused GQA attention with implicit KV repetition (no explicit expand!)
        attn_output = self.gqa_attention.gqa_attention_hip(
            query_states, key_states, value_states,
            cos, sin, self.num_kv_heads, self.softmax_scale
        )

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output

# Llama 3 70B style configuration
def get_inputs():
    return [torch.randn(4, 2048, 4096).cuda()]

def get_init_inputs():
    return [
        4096,      # hidden_size
        32,        # num_attention_heads
        8,         # num_key_value_heads
        128,       # head_dim
        4096,      # max_position_embeddings
        10000.0,   # rope_theta
        0.0,       # attention_dropout
    ]