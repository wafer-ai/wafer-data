import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

os.environ["CXX"] = "hipcc"

from torch.utils.cpp_extension import load_inline

# Fused RoPE kernel - applies rotary embeddings efficiently
fused_rope_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void fused_rope_kernel(
    const float* __restrict__ input,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_heads * seq_len * head_dim;
    
    if (idx >= total_elements) return;
    
    // Decode indices: [batch, heads, seq, head_dim]
    int d = idx % head_dim;
    int s = (idx / head_dim) % seq_len;
    int h = (idx / (head_dim * seq_len)) % num_heads;
    int b = idx / (head_dim * seq_len * num_heads);
    
    int half_dim = head_dim / 2;
    
    // cos/sin are [1, 1, seq_len, head_dim]
    int cos_sin_idx = s * head_dim + d;
    float cos_val = cos[cos_sin_idx];
    float sin_val = sin[cos_sin_idx];
    
    float x = input[idx];
    float x_rotated;
    
    if (d < half_dim) {
        // For first half: use negative of second half
        int other_idx = b * num_heads * seq_len * head_dim + 
                        h * seq_len * head_dim + 
                        s * head_dim + 
                        (d + half_dim);
        x_rotated = -input[other_idx];
    } else {
        // For second half: use first half
        int other_idx = b * num_heads * seq_len * head_dim + 
                        h * seq_len * head_dim + 
                        s * head_dim + 
                        (d - half_dim);
        x_rotated = input[other_idx];
    }
    
    output[idx] = x * cos_val + x_rotated * sin_val;
}

torch::Tensor fused_rope_hip(torch::Tensor input, torch::Tensor cos, torch::Tensor sin) {
    auto batch_size = input.size(0);
    auto num_heads = input.size(1);
    auto seq_len = input.size(2);
    auto head_dim = input.size(3);
    
    auto output = torch::empty_like(input);
    
    int total_elements = batch_size * num_heads * seq_len * head_dim;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_rope_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        cos.data_ptr<float>(),
        sin.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_heads,
        seq_len,
        head_dim
    );
    
    return output;
}

// GQA attention kernel with implicit KV repeat
__global__ void gqa_attention_kernel(
    const float* __restrict__ Q,  // [batch, num_heads, seq_len, head_dim]
    const float* __restrict__ K,  // [batch, num_kv_heads, seq_len, head_dim]
    const float* __restrict__ V,  // [batch, num_kv_heads, seq_len, head_dim]
    float* __restrict__ output,   // [batch, num_heads, seq_len, head_dim]
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    float scale
) {
    // Each block handles one (batch, head, query_pos) combination
    int b = blockIdx.z;
    int h = blockIdx.y;
    int q_pos = blockIdx.x;
    int tid = threadIdx.x;
    
    if (b >= batch_size || h >= num_heads || q_pos >= seq_len) return;
    
    // Map query head to KV head (grouped query attention)
    int kv_h = h / (num_heads / num_kv_heads);
    
    extern __shared__ float smem[];
    float* scores = smem;  // [seq_len] for attention scores
    
    // Step 1: Compute attention scores (Q * K^T) with causal mask
    float max_score = -INFINITY;
    
    // Each thread handles part of the dot product
    for (int k_pos = tid; k_pos <= q_pos; k_pos += blockDim.x) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            int q_idx = b * num_heads * seq_len * head_dim + 
                        h * seq_len * head_dim + 
                        q_pos * head_dim + d;
            int k_idx = b * num_kv_heads * seq_len * head_dim + 
                        kv_h * seq_len * head_dim + 
                        k_pos * head_dim + d;
            dot += Q[q_idx] * K[k_idx];
        }
        scores[k_pos] = dot * scale;
        max_score = fmaxf(max_score, scores[k_pos]);
    }
    
    // Fill masked positions
    for (int k_pos = q_pos + 1 + tid; k_pos < seq_len; k_pos += blockDim.x) {
        scores[k_pos] = -INFINITY;
    }
    
    __syncthreads();
    
    // Reduce max across threads
    __shared__ float shared_max;
    if (tid == 0) {
        shared_max = -INFINITY;
        for (int i = 0; i <= q_pos; i++) {
            shared_max = fmaxf(shared_max, scores[i]);
        }
    }
    __syncthreads();
    
    // Step 2: Softmax - exp and sum
    float sum_exp = 0.0f;
    for (int k_pos = tid; k_pos <= q_pos; k_pos += blockDim.x) {
        scores[k_pos] = expf(scores[k_pos] - shared_max);
        sum_exp += scores[k_pos];
    }
    
    // Reduce sum
    __shared__ float shared_sum;
    atomicAdd(&shared_sum, sum_exp);
    if (tid == 0) shared_sum = 0.0f;
    __syncthreads();
    atomicAdd(&shared_sum, sum_exp);
    __syncthreads();
    
    // Normalize
    for (int k_pos = tid; k_pos <= q_pos; k_pos += blockDim.x) {
        scores[k_pos] /= shared_sum;
    }
    for (int k_pos = q_pos + 1 + tid; k_pos < seq_len; k_pos += blockDim.x) {
        scores[k_pos] = 0.0f;
    }
    __syncthreads();
    
    // Step 3: Compute output = scores * V
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float out_val = 0.0f;
        for (int k_pos = 0; k_pos <= q_pos; k_pos++) {
            int v_idx = b * num_kv_heads * seq_len * head_dim + 
                        kv_h * seq_len * head_dim + 
                        k_pos * head_dim + d;
            out_val += scores[k_pos] * V[v_idx];
        }
        int out_idx = b * num_heads * seq_len * head_dim + 
                      h * seq_len * head_dim + 
                      q_pos * head_dim + d;
        output[out_idx] = out_val;
    }
}

torch::Tensor gqa_attention_hip(
    torch::Tensor Q, 
    torch::Tensor K, 
    torch::Tensor V,
    float scale
) {
    auto batch_size = Q.size(0);
    auto num_heads = Q.size(1);
    auto seq_len = Q.size(2);
    auto head_dim = Q.size(3);
    auto num_kv_heads = K.size(1);
    
    auto output = torch::empty_like(Q);
    
    dim3 blocks(seq_len, num_heads, batch_size);
    int threads = 128;
    size_t smem_size = seq_len * sizeof(float);
    
    gqa_attention_kernel<<<blocks, threads, smem_size>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_heads,
        num_kv_heads,
        seq_len,
        head_dim,
        scale
    );
    
    return output;
}
"""

fused_rope_cpp = """
torch::Tensor fused_rope_hip(torch::Tensor input, torch::Tensor cos, torch::Tensor sin);
torch::Tensor gqa_attention_hip(torch::Tensor Q, torch::Tensor K, torch::Tensor V, float scale);
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_rope_cpp,
    cuda_sources=fused_rope_source,
    functions=["fused_rope_hip", "gqa_attention_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
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
        
        self.fused_ops = fused_ops

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

        # Make contiguous for kernel
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        # Apply rotary embeddings using fused kernel
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        cos = cos.contiguous()
        sin = sin.contiguous()
        
        query_states = self.fused_ops.fused_rope_hip(query_states, cos, sin)
        key_states = self.fused_ops.fused_rope_hip(key_states, cos, sin)

        # Use PyTorch's efficient SDPA with GQA support (implicit KV repeat)
        # Expand KV heads to match query heads for SDPA
        # SDPA handles this efficiently internally
        key_states_expanded = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states_expanded = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Use scaled_dot_product_attention with causal mask
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states_expanded,
            value_states_expanded,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
            scale=self.softmax_scale,
        )

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output


def custom_kernel(inputs):
    hidden_states = inputs[0]
    
    # Initialize model
    batch_size = hidden_states.size(0)
    seq_len = hidden_states.size(1)
    hidden_size = hidden_states.size(2)
    
    model = ModelNew(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=4096,
    ).cuda().eval()
    
    with torch.no_grad():
        return model(hidden_states)
