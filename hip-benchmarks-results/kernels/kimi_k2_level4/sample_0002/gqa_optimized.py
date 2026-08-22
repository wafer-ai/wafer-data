import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused GQA attention kernel with RoPE, implicit KV repetition, and causal mask
gqa_attention_cpp_source = """
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

#define TILE_SIZE 64
#define WARP_SIZE 32

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
    // Block and thread indices
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    
    // Each thread block processes a tile of query positions
    int q_start = blockIdx.z * TILE_SIZE;
    int q_end = min(q_start + TILE_SIZE, seq_len);
    
    // Shared memory for Q tile (to reduce global memory access)
    __shared__ float shared_q[TILE_SIZE][64];
    
    // Shared memory for attention scores (partial)
    __shared__ float shared_scores[TILE_SIZE][TILE_SIZE];
    
    // Registers for current thread's query
    float q_vec[64];  // Max head_dim = 128, so 64 pairs
    
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    
    // Process each query position in the tile
    for (int q_pos = q_start + warp_id; q_pos < q_end; q_pos += blockDim.x / WARP_SIZE) {
        // Load query vector and apply RoPE
        // Load 2 elements per thread (x, y pair for RoPE)
        for (int d = lane * 2; d < head_dim; d += WARP_SIZE * 2) {
            int q_idx = ((batch_idx * num_heads + head_idx) * seq_len + q_pos) * head_dim + d;
            float q_x = q[q_idx];
            float q_y = q[q_idx + 1];
            
            // Apply RoPE
            int cos_sin_idx = q_pos * head_dim + d;
            float cos_val = cos[cos_sin_idx];
            float sin_val = sin[cos_sin_idx];
            
            apply_rope(q_x, q_y, cos_val, sin_val);
            
            q_vec[d] = q_x;
            q_vec[d + 1] = q_y;
        }
        
        __syncthreads();
        
        // Compute attention scores for this query position
        // For causal attention, we only compute scores for keys <= q_pos
        for (int k_pos = lane; k_pos <= q_pos; k_pos += WARP_SIZE) {
            // Load key vector, apply RoPE, and compute dot product
            float score = 0.0f;
            
            for (int d = 0; d < head_dim; d += 2) {
                int k_idx = ((batch_idx * num_kv_heads + kv_head_idx) * seq_len + k_pos) * head_dim + d;
                float k_x = k[k_idx];
                float k_y = k[k_idx + 1];
                
                // Apply RoPE to key
                int cos_sin_idx = k_pos * head_dim + d;
                float cos_val = cos[cos_sin_idx];
                float sin_val = sin[cos_sin_idx];
                
                apply_rope(k_x, k_y, cos_val, sin_val);
                
                // Accumulate dot product
                score += q_vec[d] * k_x + q_vec[d + 1] * k_y;
            }
            
            // Apply softmax scaling
            score *= softmax_scale;
            
            // Store score (causal mask already applied by loop condition)
            if (q_pos < q_end && k_pos <= q_pos) {
                shared_scores[q_pos - q_start][k_pos] = score;
            }
        }
    }
    
    __syncthreads();
    
    // Compute softmax and accumulate values
    for (int q_pos = q_start + warp_id; q_pos < q_end; q_pos += blockDim.x / WARP_SIZE) {
        // Compute softmax for this query position
        float max_score = -INFINITY;
        
        // Find max for numerical stability
        for (int k_pos = lane; k_pos <= q_pos; k_pos += WARP_SIZE) {
            float score = shared_scores[q_pos - q_start][k_pos];
            if (score > max_score) max_score = score;
        }
        
        // Warp-level reduction for max
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_max = __shfl_xor(max_score, offset);
            if (other_max > max_score) max_score = other_max;
        }
        
        // Compute sum of exponentials
        float sum_exp = 0.0f;
        for (int k_pos = lane; k_pos <= q_pos; k_pos += WARP_SIZE) {
            float score = shared_scores[q_pos - q_start][k_pos];
            float exp_score = expf(score - max_score);
            sum_exp += exp_score;
            shared_scores[q_pos - q_start][k_pos] = exp_score;  // Store for later use
        }
        
        // Warp-level reduction for sum
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum_exp += __shfl_xor(sum_exp, offset);
        }
        
        // Normalize and accumulate values
        float inv_sum_exp = 1.0f / sum_exp;
        
        // Accumulate weighted values
        for (int d = lane * 2; d < head_dim; d += WARP_SIZE * 2) {
            float out_x = 0.0f;
            float out_y = 0.0f;
            
            for (int k_pos = 0; k_pos <= q_pos; ++k_pos) {
                // Value from shared memory
                int v_idx = ((batch_idx * num_kv_heads + kv_head_idx) * seq_len + k_pos) * head_dim + d;
                float v_x = v[v_idx];
                float v_y = v[v_idx + 1];
                
                float weight = shared_scores[q_pos - q_start][k_pos] * inv_sum_exp;
                out_x += v_x * weight;
                out_y += v_y * weight;
            }
            
            // Store output
            int out_idx = ((batch_idx * num_heads + head_idx) * seq_len + q_pos) * head_dim + d;
            out[out_idx] = out_x;
            out[out_idx + 1] = out_y;
        }
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
    
    dim3 grid(batch_size, num_heads, (seq_len + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(256);  // 8 warps per block
    
    hipLaunchKernelGGL(
        gqa_attention_kernel,
        grid,
        block,
        0,
        at::hip::getCurrentHIPStreamMasqueradingAsCUDA(),
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