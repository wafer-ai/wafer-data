import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused RMSNorm kernel
rmsnorm_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void rmsnorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int hidden_size,
    float eps,
    int total_tokens
) {
    int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;
    
    const float* token_input = input + token_idx * hidden_size;
    float* token_output = output + token_idx * hidden_size;
    
    // Compute variance using parallel reduction
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = token_input[i];
        sum_sq += val * val;
    }
    
    // Warp reduction
    for (int offset = 32; offset > 0; offset /= 2) {
        sum_sq += __shfl_down(sum_sq, offset);
    }
    
    // Block reduction via shared memory
    __shared__ float shared_sum[32];
    int lane = threadIdx.x % 64;
    int warp_id = threadIdx.x / 64;
    
    if (lane == 0) {
        shared_sum[warp_id] = sum_sq;
    }
    __syncthreads();
    
    if (threadIdx.x < 32) {
        sum_sq = (threadIdx.x < (blockDim.x + 63) / 64) ? shared_sum[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_sq += __shfl_down(sum_sq, offset);
        }
    }
    
    __shared__ float rsqrt_var;
    if (threadIdx.x == 0) {
        rsqrt_var = rsqrtf(sum_sq / hidden_size + eps);
    }
    __syncthreads();
    
    // Apply normalization
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        token_output[i] = token_input[i] * rsqrt_var * weight[i];
    }
}

torch::Tensor rmsnorm_hip(torch::Tensor input, torch::Tensor weight, float eps) {
    auto output = torch::empty_like(input);
    
    int hidden_size = input.size(-1);
    int total_tokens = input.numel() / hidden_size;
    
    int block_size = 256;
    
    rmsnorm_kernel<<<total_tokens, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        hidden_size,
        eps,
        total_tokens
    );
    
    return output;
}
"""

# Fused RoPE kernel
rope_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void fused_rope_kernel(
    const float* __restrict__ q_pe,
    const float* __restrict__ k_pe,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    int batch_size,
    int num_heads,
    int seq_len,
    int rope_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = rope_dim / 2;
    int total_q = batch_size * num_heads * seq_len * rope_dim;
    int total_k = batch_size * 1 * seq_len * rope_dim;
    
    // Process Q
    if (idx < total_q) {
        int d = idx % rope_dim;
        int s = (idx / rope_dim) % seq_len;
        int h = (idx / (rope_dim * seq_len)) % num_heads;
        int b = idx / (rope_dim * seq_len * num_heads);
        
        float cos_val = cos_cache[s * rope_dim + d];
        float sin_val = sin_cache[s * rope_dim + d];
        
        int partner_d = (d < half_dim) ? (d + half_dim) : (d - half_dim);
        int partner_idx = b * num_heads * seq_len * rope_dim + h * seq_len * rope_dim + s * rope_dim + partner_d;
        
        float x = q_pe[idx];
        float x_partner = q_pe[partner_idx];
        
        float rotated;
        if (d < half_dim) {
            rotated = -x_partner;
        } else {
            rotated = x_partner;
        }
        
        q_out[idx] = x * cos_val + rotated * sin_val;
    }
    
    // Process K (single head)
    if (idx < total_k) {
        int d = idx % rope_dim;
        int s = (idx / rope_dim) % seq_len;
        int b = idx / (rope_dim * seq_len);
        
        float cos_val = cos_cache[s * rope_dim + d];
        float sin_val = sin_cache[s * rope_dim + d];
        
        int partner_d = (d < half_dim) ? (d + half_dim) : (d - half_dim);
        int partner_idx = b * seq_len * rope_dim + s * rope_dim + partner_d;
        
        float x = k_pe[idx];
        float x_partner = k_pe[partner_idx];
        
        float rotated;
        if (d < half_dim) {
            rotated = -x_partner;
        } else {
            rotated = x_partner;
        }
        
        k_out[idx] = x * cos_val + rotated * sin_val;
    }
}

std::vector<torch::Tensor> fused_rope_hip(
    torch::Tensor q_pe,
    torch::Tensor k_pe,
    torch::Tensor cos_cache,
    torch::Tensor sin_cache
) {
    auto q_out = torch::empty_like(q_pe);
    auto k_out = torch::empty_like(k_pe);
    
    int batch_size = q_pe.size(0);
    int num_heads = q_pe.size(1);
    int seq_len = q_pe.size(2);
    int rope_dim = q_pe.size(3);
    
    int total = batch_size * num_heads * seq_len * rope_dim;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    
    fused_rope_kernel<<<num_blocks, block_size>>>(
        q_pe.data_ptr<float>(),
        k_pe.data_ptr<float>(),
        cos_cache.data_ptr<float>(),
        sin_cache.data_ptr<float>(),
        q_out.data_ptr<float>(),
        k_out.data_ptr<float>(),
        batch_size,
        num_heads,
        seq_len,
        rope_dim
    );
    
    return {q_out, k_out};
}
"""

# Compile modules
rmsnorm_module = load_inline(
    name="rmsnorm_hip",
    cpp_sources="torch::Tensor rmsnorm_hip(torch::Tensor input, torch::Tensor weight, float eps);",
    cuda_sources=rmsnorm_source,
    functions=["rmsnorm_hip"],
    verbose=False,
    extra_cuda_cflags=["-O3"]
)

rope_module = load_inline(
    name="rope_hip",
    cpp_sources="std::vector<torch::Tensor> fused_rope_hip(torch::Tensor q_pe, torch::Tensor k_pe, torch::Tensor cos_cache, torch::Tensor sin_cache);",
    cuda_sources=rope_source,
    functions=["fused_rope_hip"],
    verbose=False,
    extra_cuda_cflags=["-O3"]
)


class DeepSeekRMSNormOptimized(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return rmsnorm_module.rmsnorm_hip(hidden_states.contiguous(), self.weight, self.variance_epsilon)


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


class ModelNew(nn.Module):
    """
    Optimized DeepSeek-V3 Multi-head Latent Attention (MLA)
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
        self.q_a_layernorm = DeepSeekRMSNormOptimized(q_lora_rank)
        self.q_b_proj = nn.Linear(q_lora_rank, num_attention_heads * self.q_head_dim, bias=False)

        # KV projection with LoRA compression
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = DeepSeekRMSNormOptimized(kv_lora_rank)
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
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)

        # Split query into nope and rope components
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # KV projection with compression
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        # Expand compressed KV
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        kv = kv.transpose(1, 2)

        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        
        # Use fused RoPE kernel
        q_pe_rope, k_pe_rope = rope_module.fused_rope_hip(
            q_pe.contiguous(),
            k_pe.contiguous(),
            cos.contiguous(),
            sin.contiguous()
        )

        # Assemble full query and key states
        query_states = torch.cat([q_nope, q_pe_rope], dim=-1)
        
        # Expand k_pe to all heads
        k_pe_expanded = k_pe_rope.expand(-1, self.num_heads, -1, -1)
        key_states = torch.cat([k_nope, k_pe_expanded], dim=-1)

        # Compute attention with scaled dot-product
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(q_len, q_len, device=hidden_states.device, dtype=torch.bool),
            diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output


def custom_kernel(inputs):
    # Initialize model configuration
    hidden_size = 2048
    num_attention_heads = 16
    q_lora_rank = 1536
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    max_position_embeddings = 4096
    
    model = ModelNew(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        max_position_embeddings=max_position_embeddings,
    ).cuda().eval()
    
    with torch.no_grad():
        return model(inputs[0])
