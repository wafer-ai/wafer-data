import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Streamlined GQA attention kernel with online softmax
gqa_attention_cpp_source = """
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

#define WARP_SIZE 32
#define MAX_HEAD_DIM 128
#define NEG_INF (-1e20f)

// Helper function to compute rotary embedding
__device__ __forceinline__ void apply_rope(float& x, float& y, float cos_val, float sin_val) {
    float x_new = x * cos_val - y * sin_val;
    float y_new = x * sin_val + y * cos_val;
    x = x_new;
    y = y_new;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_val = __shfl_down(val, offset);
        val = (other_val > val) ? other_val : val;
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__global__ void gqa_attention_kernel(
    const float* __restrict__ q,      // [batch, num_heads, seq_len, head_dim]
    const float* __restrict__ k,      // [batch, num_kv_heads, seq_len, head_dim]
    const float* __restrict__ v,      // [batch, num_kv_heads, seq_len, head_dim]
    float* __restrict__ out,          // [batch, num_heads, seq_len, head_dim]
    const float* __restrict__ cos,    // [1, 1, seq_len, head_dim]
    const float* __restrict__ sin,    // [1, 1, seq_len, head_dim]
    int num_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    float softmax_scale
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int q_pos = blockIdx.z;
    
    // Map query head to KV head (implicit repetition)
    int group_size = num_heads / num_kv_heads;
    int kv_head_idx = head_idx / group_size;
    
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    
    // Load query vector into registers with RoPE
    float q_vec[MAX_HEAD_DIM];
    for (int d = tid; d < head_dim; d += blockDim.x) {
        int q_idx = ((batch_idx * num_heads + head_idx) * seq_len + q_pos) * head_dim + d;
        q_vec[d] = q[q_idx];
        
        // Apply RoPE to pairs
        if (d % 2 == 0 && d + 1 < head_dim) {
            int cos_sin_idx = q_pos * head_dim + d;
            apply_rope(q_vec[d], q_vec[d + 1], cos[cos_sin_idx], sin[cos_sin_idx]);
        }
    }
    __syncthreads();
    
    // Online softmax with attention computation
    float running_max = NEG_INF;
    float running_sum = 0.0f;
    
    // Accumulate output vector
    float out_vec[MAX_HEAD_DIM];
    
    // Initialize output vector to zero
    for (int d = tid; d < head_dim; d += blockDim.x) {
        out_vec[d] = 0.0f;
    }
    __syncthreads();
    
    // Compute attention and accumulate values (single pass over keys)
    // Keys: 0 to q_pos (causal mask)
    for (int k_pos = tid; k_pos <= q_pos; k_pos += blockDim.x) {
        // Compute Q*K score
        float score = 0.0f;
        
        // Load key and apply RoPE, then compute dot product
        for (int d = 0; d < head_dim; d++) {
            int k_idx = ((batch_idx * num_kv_heads + kv_head_idx) * seq_len + k_pos) * head_dim + d;
            float k_val = k[k_idx];
            
            // Apply RoPE to key pairs
            if (d % 2 == 0 && d + 1 < head_dim) {
                float k_val2 = k[k_idx + 1];
                int cos_sin_idx = k_pos * head_dim + d;
                apply_rope(k_val, k_val2, cos[cos_sin_idx], sin[cos_sin_idx]);
                score += q_vec[d] * k_val + q_vec[d + 1] * k_val2;
            } else if (d % 2 == 0) {
                score += q_vec[d] * k_val;
            }
        }
        
        score *= softmax_scale;  // Apply scale
        
        // Update running max and sum (online softmax)
        float old_max = running_max;
        running_max = fmaxf(running_max, score);
        float exp_old = expf(old_max - running_max);
        float exp_new = expf(score - running_max);
        running_sum = running_sum * exp_old + exp_new;
        
        // Load value and accumulate weighted sum
        for (int d = tid; d < head_dim; d += blockDim.x) {
            int v_idx = ((batch_idx * num_kv_heads + kv_head_idx) * seq_len + k_pos) * head_dim + d;
            float v_val = v[v_idx];
            
            // Recompute exp weight with updated max (more stable)
            // This is approximate but stable for single-pass
            float weight = expf(score - running_max);
            // Use proper normalization at the end
            
            out_vec[d] += v_val * weight;
        }
    }
    
    __syncthreads();
    
    // Warp-level reduction for running_max and running_sum
    // First, find max across warp
    float warp_max = warp_reduce_max(running_max);
    
    // Broadcast warp_max to all threads in warp
    warp_max = __shfl(warp_max, 0);
    
    // Adjust running_sum based on warp_max
    running_sum *= expf(running_max - warp_max);
    
    __syncthreads();
    
    // Now reduce across warps using shared memory
    __shared__ float shared_max_sum[8];  // 8 warps max
    __shared__ float shared_sum_sum[8];  // 8 warps sum
    
    // Each warp writes its reduction result
    if (lane == 0) {
        shared_max_sum[warp_id] = warp_max;
        shared_sum_sum[warp_id] = running_sum;
    }
    __syncthreads();
    
    // First warp does final reduction
    float block_max = warp_max;  // Assume only one warp is active
    float block_sum = 0.0f;
    
    if (warp_id == 0 && tid < num_warps) {
        block_max = shared_max_sum[0];
        block_sum = shared_sum_sum[0];
        
        for (int i = 1; i < num_warps; i++) {
            float other_max = shared_max_sum[i];
            float other_sum = shared_sum_sum[i];
            
            if (other_max > block_max) {
                // Propagate to new max
                block_sum = block_sum * expf(block_max - other_max) + other_sum;
                block_max = other_max;
            } else {
                // Keep old max
                block_sum += other_sum * expf(other_max - block_max);
            }
        }
        
        // Store final result
        shared_max_sum[0] = block_max;
        shared_sum_sum[0] = block_sum;
    }
    __syncthreads();
    
    block_max = shared_max_sum[0];
    block_sum = shared_sum_sum[0];
    
    // Normalize output
    float inv_sum = 1.0f / fmaxf(block_sum, 1e-30f);
    
    // Store final output
    for (int d = tid; d < head_dim; d += blockDim.x) {
        int out_idx = ((batch_idx * num_heads + head_idx) * seq_len + q_pos) * head_dim + d;
        float normalized_val = out_vec[d] * expf(running_max - block_max) * inv_sum;
        out[out_idx] = normalized_val;
    }
}

torch::Tensor gqa_attention_hip(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor cos, torch::Tensor sin,
    int num_kv_heads, float softmax_scale
) {
    auto batch_size = q.size(0);
    auto num_heads = q.size(1);
    auto seq_len = q.size(2);
    auto head_dim = q.size(3);
    
    auto out = torch::zeros_like(q);
    
    // Launch: one block per (batch, head, query position)
    dim3 grid(batch_size, num_heads, seq_len);
    dim3 block(256);  // 256 threads = 8 warps
    
    gqa_attention_kernel<<<grid, block, 0>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        cos.data_ptr<float>(),
        sin.data_ptr<float>(),
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