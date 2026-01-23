import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Conservative GQA kernel with PyTorch-style implementation in HIP
gqa_attention_cpp_source = """
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

#define WARP_SIZE 32
#define MAX_HEAD_DIM 128

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
    
    // Map query head to KV head (implicit repetition)
    int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    
    int tid = threadIdx.x;
    
    // Shared memory for reduction (max + sum)
    extern __shared__ float shared_mem[];
    float* shared_max = shared_mem;
    float* shared_sum = &shared_mem[1];
    
    // Load query vector in registers and apply RoPE
    float q_vec[MAX_HEAD_DIM];
    
    for (int d = tid; d < head_dim; d += blockDim.x) {
        int q_idx = ((batch_idx * num_heads + head_idx) * seq_len + q_pos) * head_dim + d;
        q_vec[d] = q[q_idx];
        
        // Apply RoPE in pairs
        if (d % 2 == 0 && d + 1 < head_dim) {
            int cos_sin_idx = q_pos * head_dim + d;
            apply_rope(q_vec[d], q_vec[d + 1], cos[cos_sin_idx], sin[cos_sin_idx]);
        }
    }
    
    __syncthreads();
    
    // Compute attention scores for all keys up to q_pos (causal)
    float thread_max = -1e20f;
    float thread_sum = 0.0f;
    
    for (int k_pos = tid; k_pos <= q_pos; k_pos += blockDim.x) {
        // Compute dot product between q and k
        float score = 0.0f;
        
        for (int d = 0; d < head_dim; d++) {
            int k_idx = ((batch_idx * num_kv_heads + kv_head_idx) * seq_len + k_pos) * head_dim + d;
            float k_val = k[k_idx];
            
            // Apply RoPE to key if it's a pair
            if (d % 2 == 0 && d + 1 < head_dim) {
                float k_val2 = k[k_idx + 1];
                int cos_sin_idx = k_pos * head_dim + d;
                apply_rope(k_val, k_val2, cos[cos_sin_idx], sin[cos_sin_idx]);
                score += q_vec[d] * k_val + q_vec[d + 1] * k_val2;
            } else if (d % 2 == 0) {
                score += q_vec[d] * k_val;
            }
        }
        
        // Apply softmax scaling
        score *= softmax_scale;
        
        // Track max for numerical stability
        thread_max = fmaxf(thread_max, score);
    }
    
    // Block-level reduction for max
    // Store in shared memory
    if (tid == 0) shared_max[0] = -1e20f;
    __syncthreads();
    
    // Thread writes its max
    atomicMaxFloat(shared_max, thread_max);
    __syncthreads();
    
    float global_max = shared_max[0];
    
    // Compute exp scores and sum
    for (int k_pos = tid; k_pos <= q_pos; k_pos += blockDim.x) {
        // Recompute score
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            int k_idx = ((batch_idx * num_kv_heads + kv_head_idx) * seq_len + k_pos) * head_dim + d;
            float k_val = k[k_idx];
            
            if (d % 2 == 0 && d + 1 < head_dim) {
                float k_val2 = k[k_idx + 1];
                int cos_sin_idx = k_pos * head_dim + d;
                apply_rope(k_val, k_val2, cos[cos_sin_idx], sin[cos_sin_idx]);
                score += q_vec[d] * k_val + q_vec[d + 1] * k_val2;
            } else if (d % 2 == 0) {
                score += q_vec[d] * k_val;
            }
        }
        
        score *= softmax_scale;
        
        // Compute exp and add to sum
        float exp_score = expf(score - global_max);
        thread_sum += exp_score;
    }
    
    // Block-level reduction for sum
    if (tid == 0) shared_sum[0] = 0.0f;
    __syncthreads();
    
    // Add thread sums
    atomicAdd(shared_sum, thread_sum);
    __syncthreads();
    
    float global_sum = shared_sum[0];
    
    // Compute output attention
    float inv_sum = 1.0f / global_sum;
    
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float out_val = 0.0f;
        
        for (int k_pos = 0; k_pos <= q_pos; k_pos++) {
            // Compute attention weight
            float score = 0.0f;
            for (int d2 = 0; d2 < head_dim; d2++) {
                int k_idx = ((batch_idx * num_kv_heads + kv_head_idx) * seq_len + k_pos) * head_dim + d2;
                float k_val = k[k_idx];
                
                if (d2 % 2 == 0 && d2 + 1 < head_dim) {
                    float k_val2 = k[k_idx + 1];
                    int cos_sin_idx = k_pos * head_dim + d2;
                    apply_rope(k_val, k_val2, cos[cos_sin_idx], sin[cos_sin_idx]);
                    score += q_vec[d2] * k_val + q_vec[d2 + 1] * k_val2;
                } else if (d2 % 2 == 0) {
                    score += q_vec[d2] * k_val;
                }
            }
            score *= softmax_scale;
            float weight = expf(score - global_max) * inv_sum;
            
            // Load value and accumulate
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
    auto batch_size = q.size(0);
    auto num_heads = q.size(1);
    auto seq_len = q.size(2);
    auto head_dim = q.size(3);
    
    auto out = torch::zeros_like(q);
    
    dim3 grid(batch_size, num_heads, seq_len);
    dim3 block(256);
    size_t shared_mem_size = 2 * sizeof(float);  // For max and sum
    
    gqa_attention_kernel<<<grid, block, shared_mem_size>>>(
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