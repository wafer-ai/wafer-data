import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Final correct GQA attention kernel
gqa_attention_cpp_source = """
#include <hip/hip_runtime.h>
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

// Online softmax approach for numerical stability
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
    
    // Map query head to KV head (implicit repetition)
    int group_size = num_heads / num_kv_heads;
    int kv_head_idx = head_idx / group_size;
    
    int tid = threadIdx.x;
    
    // Load query vector into registers with RoPE
    __shared__ float shared_q[MAX_HEAD_DIM];
    
    for (int d = tid; d < head_dim; d += blockDim.x) {
        int q_idx = ((batch_idx * num_heads + head_idx) * seq_len + q_pos) * head_dim + d;
        shared_q[d] = q[q_idx];
        
        // Apply RoPE in pairs
        if (d % 2 == 0 && d + 1 < head_dim) {
            int cos_sin_idx = q_pos * head_dim + d;
            apply_rope(shared_q[d], shared_q[d + 1], cos[cos_sin_idx], sin[cos_sin_idx]);
        }
    }
    __syncthreads();
    
    // Online softmax: track running max and sum
    float running_max = NEG_INF;
    float running_sum = 0.0f;
    
    // Accumulate weighted values
    __shared__ float out_buf[MAX_HEAD_DIM];
    for (int d = tid; d < head_dim; d += blockDim.x) {
        out_buf[d] = 0.0f;
    }
    __syncthreads();
    
    // Single-pass over keys (causal: keys = 0 to q_pos)
    for (int k_pos = tid; k_pos <= q_pos; k_pos += blockDim.x) {
        // Compute Q*K dot product
        float score = 0.0f;
        
        // Load and apply RoPE to key, compute dot product
        for (int d = 0; d < head_dim; d += 2) {
            int k_idx = ((batch_idx * num_kv_heads + kv_head_idx) * seq_len + k_pos) * head_dim + d;
            float k_x = k[k_idx];
            float k_y = k[k_idx + 1];
            
            if (d + 1 < head_dim) {
                // Apply RoPE to key
                int cos_sin_idx = k_pos * head_dim + d;
                apply_rope(k_x, k_y, cos[cos_sin_idx], sin[cos_sin_idx]);
                
                // Dot product with query
                score += shared_q[d] * k_x + shared_q[d + 1] * k_y;
            } else {
                score += shared_q[d] * k_x;
            }
        }
        
        // Apply softmax scaling
        score *= softmax_scale;
        
        // Online softmax update
        float old_max = running_max;
        running_max = fmaxf(running_max, score);
        
        float exp_old = expf(old_max - running_max);
        float exp_new = expf(score - running_max);
        
        running_sum = running_sum * exp_old + exp_new;
        
        // Load value and accumulate weighted sum
        for (int d = tid; d < head_dim; d += blockDim.x) {
            int v_idx = ((batch_idx * num_kv_heads + kv_head_idx) * seq_len + k_pos) * head_dim + d;
            float v_val = v[v_idx];
            
            // Normalize and accumulate
            // Note: we adjust by exp(old_max - running_max) when max changes
            if (running_max > old_max && running_max > NEG_INF) {
                float old_weight = expf(old_max - running_max);
                atomicMulFloat(&out_buf[d], old_weight);
            }
            
            // Add current weighted value
            float weight = exp_new / fmaxf(running_sum, 1e-30f);
            atomicAdd(&out_buf[d], v_val * exp_new);
        }
    }
    __syncthreads();
    
    // Normalize and write output
    float inv_sum = 1.0f / fmaxf(running_sum, 1e-30f);
    
    for (int d = tid; d < head_dim; d += blockDim.x) {
        int out_idx = ((batch_idx * num_heads + head_idx) * seq_len + q_pos) * head_dim + d;
        out[out_idx] = out_buf[d] * inv_sum * expf(running_max - running_max);
    }
}

// Simple atomic operations helper
__device__ __forceinline__ void atomicAdd(float* address, float val) {
    atomicAdd_block(address, val);
}

__device__ __forceinline__ void atomicMaxFloat(float* address, float val) {
    float old = *address;
    float assumed;
    do {
        assumed = old;
        old = atomicCAS_block((int*)address, __float_as_int(assumed), __float_as_int(fmaxf(val, assumed)));
    } while (assumed != old);
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
    
    dim3 grid(batch_size, num_heads, seq_len);
    dim3 block(256);  // 256 threads per block
    
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