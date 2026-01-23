import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Corrected GQA attention kernel with proper warp reductions
gqa_attention_cpp_source = """
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

#define WARP_SIZE 32
#define MAX_HEAD_DIM 128
#define INF 1e20f

// Helper function to compute rotary embedding
__device__ __forceinline__ void apply_rope(float& x, float& y, float cos_val, float sin_val) {
    float x_new = x * cos_val - y * sin_val;
    float y_new = x * sin_val + y * cos_val;
    x = x_new;
    y = y_new;
}

// Warp reduction for max
__device__ __forceinline__ float warp_reduce_max(float val, unsigned int mask = 0xffffffff) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down(val, offset));
    }
    return val;
}

// Warp reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val, unsigned int mask = 0xffffffff) {
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
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    float softmax_scale
) {
    // Each thread block handles: batch element, one head, one query position
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int q_pos = blockIdx.z;
    
    int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    
    // Use registers for query vector (max head_dim = 128)
    float q_vec[MAX_HEAD_DIM];
    
    // Load query vector and apply RoPE
    for (int d = tid; d < head_dim; d += blockDim.x) {
        int q_idx = ((batch_idx * num_heads + head_idx) * seq_len + q_pos) * head_dim + d;
        float val = q[q_idx];
        
        // Apply RoPE in pairs
        if (d % 2 == 0 && d + 1 < head_dim) {
            float val2 = q[q_idx + 1];
            int cos_sin_idx = q_pos * head_dim + d;
            apply_rope(val, val2, cos[cos_sin_idx], sin[cos_sin_idx]);
            q_vec[d] = val;
            q_vec[d + 1] = val2;
        } else if (d % 2 == 0) {
            q_vec[d] = val;
        }
    }
    
    __syncthreads();
    
    // Each thread computes scores for a subset of keys
    int num_keys = q_pos + 1;  // Causal mask
    int keys_per_thread = (num_keys + blockDim.x - 1) / blockDim.x;
    
    float thread_max = -INF;
    float thread_sum = 0.0f;
    
    // Compute scores and find max
    for (int i = 0; i < keys_per_thread; i++) {
        int k_pos = i * blockDim.x + tid;
        if (k_pos < num_keys) {
            // Compute dot product
            float score = 0.0f;
            
            for (int d = 0; d < head_dim; d += 2) {
                int k_idx = ((batch_idx * num_kv_heads + kv_head_idx) * seq_len + k_pos) * head_dim + d;
                float k_x = k[k_idx];
                float k_y = k[k_idx + 1];
                
                // Apply RoPE to key
                if (d + 1 < head_dim) {
                    int cos_sin_idx = k_pos * head_dim + d;
                    apply_rope(k_x, k_y, cos[cos_sin_idx], sin[cos_sin_idx]);
                }
                
                // Accumulate dot product
                score += q_vec[d] * k_x + (d + 1 < head_dim ? q_vec[d + 1] * k_y : 0.0f);
            }
            
            // Apply softmax scaling
            score *= softmax_scale;
            
            // Update thread max
            thread_max = fmaxf(thread_max, score);
        }
    }
    
    // Warp-level reduction for max
    float warp_max = warp_reduce_max(thread_max);
    
    // Broadcast max to all threads
    if (lane == 0) {
        // Use shared memory to broadcast
        extern __shared__ float shared_max[];
        shared_max[warp_id] = warp_max;
    }
    __syncthreads();
    
    float global_max = -INF;
    for (int i = tid; i < blockDim.x / WARP_SIZE; i += blockDim.x) {
        extern __shared__ float shared_max[];
        if (i < blockDim.x / WARP_SIZE) {
            global_max = fmaxf(global_max, shared_max[i]);
        }
    }
    __syncthreads();
    
    // Broadcast to all threads
    extern __shared__ float shared_max[];
    if (tid == 0) {
        for (int i = 0; i < blockDim.x / WARP_SIZE; i++) {
            global_max = fmaxf(global_max, shared_max[i]);
        }
        shared_max[0] = global_max;
    }
    __syncthreads();
    global_max = shared_max[0];
    
    // Compute exp and sum
    thread_sum = 0.0f;
    for (int i = 0; i < keys_per_thread; i++) {
        int k_pos = i * blockDim.x + tid;
        if (k_pos < num_keys) {
            // Compute dot product again (recompute to avoid storing all scores)
            float score = 0.0f;
            
            for (int d = 0; d < head_dim; d += 2) {
                int k_idx = ((batch_idx * num_kv_heads + kv_head_idx) * seq_len + k_pos) * head_dim + d;
                float k_x = k[k_idx];
                float k_y = k[k_idx + 1];
                
                // Apply RoPE to key
                if (d + 1 < head_dim) {
                    int cos_sin_idx = k_pos * head_dim + d;
                    apply_rope(k_x, k_y, cos[cos_sin_idx], sin[cos_sin_idx]);
                }
                
                // Accumulate dot product
                score += q_vec[d] * k_x + (d + 1 < head_dim ? q_vec[d + 1] * k_y : 0.0f);
            }
            
            score *= softmax_scale;
            float exp_val = expf(score - global_max);
            thread_sum += exp_val;
        }
    }
    
    // Block-level reduction for sum
    float block_sum = 0.0f;
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down(thread_sum, offset);
    }
    if (lane == 0) {
        extern __shared__ float shared_sum[];
        shared_sum[warp_id] = thread_sum;
    }
    __syncthreads();
    
    if (tid < blockDim.x / WARP_SIZE) {
        extern __shared__ float shared_sum[];
        block_sum = shared_sum[tid];
    }
    __syncthreads();
    
    if (tid == 0) {
        extern __shared__ float shared_sum[];
        for (int i = 0; i < blockDim.x / WARP_SIZE; i++) {
            block_sum += shared_sum[i];
        }
        shared_sum[0] = block_sum;
    }
    __syncthreads();
    block_sum = shared_sum[0];
    
    // Normalize and accumulate values
    float inv_sum = 1.0f / fmaxf(block_sum, 1e-30f);
    
    // Each thread accumulates part of the output
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float out_val = 0.0f;
        
        for (int i = 0; i < keys_per_thread; i++) {
            int k_pos = i * blockDim.x + tid;
            if (k_pos < num_keys) {
                // Compute attention weight
                float score = 0.0f;
                for (int d2 = 0; d2 < head_dim; d2 += 2) {
                    int k_idx = ((batch_idx * num_kv_heads + kv_head_idx) * seq_len + k_pos) * head_dim + d2;
                    float k_x = k[k_idx];
                    float k_y = k[k_idx + 1];
                    
                    if (d2 + 1 < head_dim) {
                        int cos_sin_idx = k_pos * head_dim + d2;
                        apply_rope(k_x, k_y, cos[cos_sin_idx], sin[cos_sin_idx]);
                    }
                    
                    score += q_vec[d2] * k_x + (d2 + 1 < head_dim ? q_vec[d2 + 1] * k_y : 0.0f);
                }
                score *= softmax_scale;
                float weight = expf(score - global_max) * inv_sum;
                
                // Accumulate weighted value
                int v_idx = ((batch_idx * num_kv_heads + kv_head_idx) * seq_len + k_pos) * head_dim + d;
                float v_val = v[v_idx];
                out_val += v_val * weight;
            }
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
    
    // Launch configuration: one block per (batch, head, query position)
    dim3 grid(batch_size, num_heads, seq_len);
    dim3 block(256);  // 256 threads per block
    
    // Call kernel with shared memory: 1 float per warp for max + 1 per warp for sum
    gqa_attention_kernel<<<grid, block, (block.x / WARP_SIZE) * 2 * sizeof(float)>>>(
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