import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused attention kernel (QK matmul + causal mask + softmax + V matmul)
fused_attention_cpp_source = """
#include <hip/hip_runtime.h>
#include <cmath>
#include <float.h>

__global__ void fused_attention_kernel(
    const float* query_states,
    const float* key_states,
    const float* value_states,
    float* attn_output,
    int bsz,
    int num_heads,
    int q_len,
    int q_head_dim,
    int v_head_dim,
    float softmax_scale
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int q_idx = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (q_idx >= q_len) return;
    
    __shared__ float attn_scores[2048];
    __shared__ float max_score;
    __shared__ float exp_sum;
    
    // Each thread block computes attention for one batch and one head
    // Thread (x, y) computes for query position q_idx and value dimension threadIdx.y
    
    // Step 1: Compute attention scores and find max for numerical stability
    if (threadIdx.y == 0) {
        float local_max = -FLT_MAX;
        
        for (int k_idx = 0; k_idx < q_len; k_idx++) {
            float score = 0.0f;
            
            // Only compute for causal positions (k_idx <= q_idx)
            if (k_idx <= q_idx) {
                // Dot product: query[batch_idx, head_idx, q_idx] · key[batch_idx, head_idx, k_idx]
                for (int d = 0; d < q_head_dim; d++) {
                    float q_val = query_states[batch_idx * num_heads * q_len * q_head_dim +
                                              head_idx * q_len * q_head_dim +
                                              q_idx * q_head_dim + d];
                    float k_val = key_states[batch_idx * num_heads * q_len * q_head_dim +
                                            head_idx * q_len * q_head_dim +
                                            k_idx * q_head_dim + d];
                    score += q_val * k_val;
                }
                score *= softmax_scale;
            } else {
                score = -FLT_MAX;  // Apply causal mask
            }
            
            attn_scores[k_idx] = score;
            if (score > local_max) local_max = score;
        }
        
        max_score = local_max;
    }
    
    __syncthreads();
    
    // Step 2: Compute softmax (only thread 0 of each warp)
    if (threadIdx.y == 0) {
        float sum_exp = 0.0f;
        
        for (int k_idx = 0; k_idx < q_len; k_idx++) {
            if (attn_scores[k_idx] > -FLT_MAX/2) {
                float exp_score = expf(attn_scores[k_idx] - max_score);
                attn_scores[k_idx] = exp_score;
                sum_exp += exp_score;
            } else {
                attn_scores[k_idx] = 0.0f;
            }
        }
        
        exp_sum = sum_exp;
        
        // Normalize to get attention weights
        for (int k_idx = 0; k_idx < q_len; k_idx++) {
            attn_scores[k_idx] /= sum_exp;
        }
    }
    
    __syncthreads();
    
    // Step 3: Compute attention output (weighted sum of values)
    // Each thread computes one element of the output: attn_output[batch_idx, head_idx, q_idx, v_d]
    for (int v_d = threadIdx.y; v_d < v_head_dim; v_d += blockDim.y) {
        float out_val = 0.0f;
        
        for (int k_idx = 0; k_idx < q_len; k_idx++) {
            float weight = attn_scores[k_idx];
            if (weight > 0.0f) {
                float v_val = value_states[batch_idx * num_heads * q_len * v_head_dim +
                                          head_idx * q_len * v_head_dim +
                                          k_idx * v_head_dim + v_d];
                out_val += weight * v_val;
            }
        }
        
        attn_output[batch_idx * num_heads * q_len * v_head_dim +
                   head_idx * q_len * v_head_dim +
                   q_idx * v_head_dim + v_d] = out_val;
    }
}

torch::Tensor fused_attention_hip(
    torch::Tensor query_states,
    torch::Tensor key_states,
    torch::Tensor value_states,
    float softmax_scale
) {
    auto bsz = query_states.size(0);
    auto num_heads = query_states.size(1);
    auto q_len = query_states.size(2);
    auto q_head_dim = query_states.size(3);
    auto v_head_dim = value_states.size(3);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(query_states.device());
    auto attn_output = torch::zeros({bsz, num_heads, q_len, v_head_dim}, options);
    
    dim3 grid(bsz, num_heads, (q_len + 63) / 64);
    dim3 block(64, 16);
    
    fused_attention_kernel<<<grid, block>>>(
        query_states.data_ptr<float>(),
        key_states.data_ptr<float>(),
        value_states.data_ptr<float>(),
        attn_output.data_ptr<float>(),
        bsz, num_heads, q_len, q_head_dim, v_head_dim, softmax_scale
    );
    
    return attn_output;
}
"""

# Compile the fused attention kernel
fused_attention = load_inline(
    name="fused_attention",
    cpp_sources=fused_attention_cpp_source,
    functions=["fused_attention_hip"],
    verbose=True,
)

class DeepSeekRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class DeepSeekRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000.0):
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
        return emb.cos(), emb.sin()

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # Patched for [bs, heads, seq, dim] layout: [seq, dim] -> [1, 1, seq, dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class ModelNew(nn.Module):
    """
    DeepSeek-V3 Multi-head Latent Attention (MLA) with fused attention kernel
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.attention_dropout = attention_dropout
        self.softmax_scale = self.q_head_dim ** (-0.5)

        # Query projection with LoRA compression
        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_a_layernorm = DeepSeekRMSNorm(q_lora_rank)
        self.q_b_proj = nn.Linear(q_lora_rank, num_attention_heads * self.q_head_dim, bias=False)

        # KV projection with LoRA compression
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = DeepSeekRMSNorm(kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            num_attention_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )

        # Output projection
        self.o_proj = nn.Linear(num_attention_heads * v_head_dim, hidden_size, bias=False)

        # Rotary embeddings
        self.rotary_emb = DeepSeekRotaryEmbedding(
            qk_rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Query projection with LoRA compression
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        query_states = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)

        # KV projection with compression
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        kv = kv.transpose(1, 2)

        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Rotary embeddings
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        q_pe = query_states[:, :, :, self.qk_nope_head_dim:]
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin)
        
        # Assemble full query and key states
        query_states = torch.cat([
            query_states[:, :, :, :self.qk_nope_head_dim],
            q_pe
        ], dim=-1)
        
        k_pe_broadcast = k_pe.expand(-1, self.num_heads, -1, -1)
        key_states = torch.cat([k_nope, k_pe_broadcast], dim=-1)

        # Fused attention computation (replaces 3 PyTorch ops with 1 kernel)
        attn_output = fused_attention.fused_attention_hip(
            query_states,
            key_states,
            value_states,
            self.softmax_scale
        )

        # Reshape and output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output

# DeepSeek-V3 style configuration (scaled down for MI300X)
batch_size = 4
seq_len = 2048
hidden_size = 2048
num_attention_heads = 16
q_lora_rank = 1536
kv_lora_rank = 512
qk_nope_head_dim = 128
qk_rope_head_dim = 64
v_head_dim = 128
max_position_embeddings = 4096

def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size).cuda()]

def get_init_inputs():
    return [
        hidden_size,
        num_attention_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        max_position_embeddings,
    ]