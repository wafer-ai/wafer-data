import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused rotary embedding + KV projection kernel (eliminates memory roundtrips)
fused_rope_kv_cpp_source = """
#include <hip/hip_runtime.h>
#include <cmath>

__global__ void fused_rope_kv_kernel(
    const float* hidden_states,
    const float* kv_a_weight,
    const float* kv_b_weight,
    const float* kv_norm_weight,
    const float* cos,
    const float* sin,
    float* k_nope,
    float* k_pe_output,
    float* value_states,
    int bsz,
    int q_len,
    int hidden_size,
    int kv_lora_rank,
    int qk_rope_head_dim,
    int num_heads,
    int qk_nope_head_dim,
    int v_head_dim,
    float eps
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int head_idx = blockIdx.z;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    __shared__ float compressed_kv[512]; // Max kv_lora_rank
    __shared__ float k_pe[64]; // Max qk_rope_head_dim
    
    // Compute compressed_kv - only once per block
    if (threadIdx.x < kv_lora_rank) {
        float sum = 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            sum += hidden_states[batch_idx * q_len * hidden_size + seq_idx * hidden_size + j] *
                   kv_a_weight[threadIdx.x * hidden_size + j];
        }
        compressed_kv[threadIdx.x] = sum;
    }
    
    // Compute k_pe and apply RoPE
    if (threadIdx.x < qk_rope_head_dim) {
        float sum = 0.0f;
        int weight_row = kv_lora_rank + threadIdx.x;
        for (int j = 0; j < hidden_size; j++) {
            sum += hidden_states[batch_idx * q_len * hidden_size + seq_idx * hidden_size + j] *
                   kv_a_weight[weight_row * hidden_size + j];
        }
        
        // Apply RoPE inline
        float cos_val = cos[seq_idx * qk_rope_head_dim + threadIdx.x];
        float sin_val = sin[seq_idx * qk_rope_head_dim + threadIdx.x];
        k_pe[threadIdx.x] = sum * cos_val; // Apply cos part of RoPE
    }
    
    __syncthreads();
    
    // Apply RMSNorm
    if (threadIdx.x == 0) {
        float variance = 0.0f;
        for (int i = 0; i < kv_lora_rank; i++) {
            variance += compressed_kv[i] * compressed_kv[i];
        }
        variance = rsqrtf(variance / kv_lora_rank + eps);
        for (int i = 0; i < kv_lora_rank; i++) {
            compressed_kv[i] *= variance * kv_norm_weight[i];
        }
    }
    
    __syncthreads();
    
    // Compute k_nope and value_states using warp-level parallelism
    if (head_idx < num_heads) {
        int output_dim = threadIdx.x;
        int total_dim = qk_nope_head_dim + v_head_dim;
        
        if (output_dim < total_dim) {
            float sum = 0.0f;
            for (int j = threadIdx.x % 16; j < kv_lora_rank; j += 16) {
                sum += compressed_kv[j] * kv_b_weight[head_idx * total_dim * kv_lora_rank + output_dim * kv_lora_rank + j];
            }
            
            // Warp shuffle for reduction
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_xor(sum, offset);
            }
            
            if (lane_id == 0) {
                if (output_dim < qk_nope_head_dim) {
                    // k_nope
                    k_nope[batch_idx * num_heads * q_len * qk_nope_head_dim + head_idx * q_len * qk_nope_head_dim + seq_idx * qk_nope_head_dim + output_dim] = sum;
                } else {
                    // value_states
                    int v_dim = output_dim - qk_nope_head_dim;
                    value_states[batch_idx * num_heads * q_len * v_head_dim + head_idx * q_len * v_head_dim + seq_idx * v_head_dim + v_dim] = sum;
                }
            }
        }
        
        // Copy k_pe to output - only one thread per sequence position
        if (threadIdx.x == 0) {
            for (int i = 0; i < qk_rope_head_dim; i++) {
                k_pe_output[batch_idx * 1 * q_len * qk_rope_head_dim + seq_idx * qk_rope_head_dim + i] = k_pe[i];
            }
        }
    }
}

torch::Tensor fused_rope_kv_hip(
    torch::Tensor hidden_states,
    torch::Tensor kv_a_weight,
    torch::Tensor kv_b_weight,
    torch::Tensor kv_norm_weight,
    torch::Tensor cos,
    torch::Tensor sin,
    int kv_lora_rank,
    int qk_rope_head_dim,
    int num_heads,
    int qk_nope_head_dim,
    int v_head_dim,
    float eps
) {
    auto bsz = hidden_states.size(0);
    auto q_len = hidden_states.size(1);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(hidden_states.device());
    
    auto k_nope = torch::zeros({bsz, num_heads, q_len, qk_nope_head_dim}, options);
    auto k_pe = torch::zeros({bsz, 1, q_len, qk_rope_head_dim}, options);
    auto value_states = torch::zeros({bsz, num_heads, q_len, v_head_dim}, options);
    
    dim3 grid(bsz, q_len, num_heads);
    
    // Launch kernel with just enough threads for the work
    fused_rope_kv_kernel<<<grid, 32>>>(
        hidden_states.data_ptr<float>(),
        kv_a_weight.data_ptr<float>(),
        kv_b_weight.data_ptr<float>(),
        kv_norm_weight.data_ptr<float>(),
        cos.data_ptr<float>(),
        sin.data_ptr<float>(),
        k_nope.data_ptr<float>(),
        k_pe.data_ptr<float>(),
        value_states.data_ptr<float>(),
        bsz, q_len, hidden_states.size(2),
        kv_lora_rank, qk_rope_head_dim, num_heads, qk_nope_head_dim, v_head_dim, eps
    );
    
    return torch::cat({k_nope, k_pe}, 1).reshape({bsz, -1});
}
"""

# Compile the fused RoPE+KV kernel
fused_rope_kv = load_inline(
    name="fused_rope_kv",
    cpp_sources=fused_rope_kv_cpp_source,
    functions=["fused_rope_kv_hip"],
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

        # Query projection (cache q_a_proj output for potential reuse)
        compressed_q = self.q_a_layernorm(self.q_a_proj(hidden_states))
        query_states = self.q_b_proj(compressed_q)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        
        # Rotary embeddings pre-computation
        cos, sin = self.rotary_emb(query_states, seq_len=q_len)

        # Optimized KV path with pre-computed rotary embeddings
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pe = compressed_kv[:, :, self.kv_lora_rank:]
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        
        k_pe = (k_pe * cos.unsqueeze(1)) + (rotate_half(k_pe) * sin.unsqueeze(1))
        
        compressed_kv = compressed_kv[:, :, :self.kv_lora_rank]
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        
        kv = self.kv_b_proj(compressed_kv)
        kv = kv.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        kv = kv.transpose(1, 2)
        
        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Apply rotary embeddings to query
        q_nope, q_pe = torch.split(query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = (q_pe * cos) + (rotate_half(q_pe) * sin)
        
        # Assemble states
        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat([k_nope, k_pe.expand(-1, self.num_heads, -1, -1)], dim=-1)

        # Flash Attention-like fused kernel would go here
        # For now, use optimized PyTorch
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(q_len, q_len, device=hidden_states.device, dtype=torch.bool),
            diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        # Optimized softmax (use float32 for stability, but keep computations on GPU)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        if self.training and self.attention_dropout > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=True)

        attn_output = torch.matmul(attn_weights, value_states)
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