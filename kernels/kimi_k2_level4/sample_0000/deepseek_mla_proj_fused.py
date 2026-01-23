import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused Q projection kernel (q_a_proj + layernorm + q_b_proj)
fused_q_proj_cpp_source = """
#include <hip/hip_runtime.h>
#include <cmath>

__global__ void fused_q_proj_kernel(
    const float* hidden_states,
    const float* q_a_weight,
    const float* q_b_weight,
    const float* q_norm_weight,
    float* q_output,
    int bsz,
    int q_len,
    int hidden_size,
    int q_lora_rank,
    int num_heads,
    int q_head_dim,
    float eps
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int head_idx = blockIdx.z;
    
    extern __shared__ float shared_mem[];
    float* compressed_q = shared_mem;
    
    // Step 1: Q compression (q_a_proj)
    for (int i = threadIdx.x; i < q_lora_rank; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            sum += hidden_states[batch_idx * q_len * hidden_size + seq_idx * hidden_size + j] * 
                   q_a_weight[i * hidden_size + j];
        }
        compressed_q[i] = sum;
    }
    
    __syncthreads();
    
    // Step 2: RMSNorm (parallel reduction)
    if (threadIdx.x == 0) {
        float variance = 0.0f;
        for (int i = 0; i < q_lora_rank; i++) {
            variance += compressed_q[i] * compressed_q[i];
        }
        variance /= q_lora_rank;
        float rms = rsqrtf(variance + eps);
        
        for (int i = 0; i < q_lora_rank; i++) {
            compressed_q[i] = compressed_q[i] * rms * q_norm_weight[i];
        }
    }
    
    __syncthreads();
    
    // Step 3: Q expansion (q_b_proj)
    for (int i = threadIdx.x; i < q_head_dim; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < q_lora_rank; j++) {
            sum += compressed_q[j] * q_b_weight[head_idx * q_head_dim * q_lora_rank + i * q_lora_rank + j];
        }
        q_output[batch_idx * num_heads * q_len * q_head_dim + head_idx * q_len * q_head_dim + seq_idx * q_head_dim + i] = sum;
    }
}

torch::Tensor fused_q_proj_hip(
    torch::Tensor hidden_states,
    torch::Tensor q_a_weight,
    torch::Tensor q_b_weight,
    torch::Tensor q_norm_weight,

int q_lora_rank,
    int num_heads,
    int q_head_dim,
    float eps
) {
    auto bsz = hidden_states.size(0);
    auto q_len = hidden_states.size(1);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(hidden_states.device());
    auto q_output = torch::zeros({bsz, num_heads, q_len, q_head_dim}, options);
    
    dim3 grid(bsz, q_len, num_heads);
    int shared_mem_size = q_lora_rank * sizeof(float);
    
    fused_q_proj_kernel<<<grid, 256, shared_mem_size>>>(
        hidden_states.data_ptr<float>(),
        q_a_weight.data_ptr<float>(),
        q_b_weight.data_ptr<float>(),
        q_norm_weight.data_ptr<float>(),
        q_output.data_ptr<float>(),
        bsz, q_len, hidden_states.size(2),
        q_lora_rank, num_heads, q_head_dim, eps
    );
    
    return q_output;
}
"""

# Fused KV projection kernel (kv_a_proj + split + layernorm + kv_b_proj)
fused_kv_proj_cpp_source = """
#include <hip/hip_runtime.h>
#include <cmath>

__global__ void fused_kv_proj_kernel(
    const float* hidden_states,
    const float* kv_a_weight,
    const float* kv_b_weight,
    const float* kv_norm_weight,
    float* k_nope,
    float* k_pe,
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
    
    extern __shared__ float shared_mem[];
    float* compressed_kv = shared_mem;
    
    // Compute k_pe (positional part) - only once per sequence position
    if (head_idx == 0 && threadIdx.x == 0) {
        for (int i = 0; i < qk_rope_head_dim; i++) {
            float sum = 0.0f;
            int weight_row = kv_lora_rank + i;
            for (int j = 0; j < hidden_size; j++) {
                sum += hidden_states[batch_idx * q_len * hidden_size + seq_idx * hidden_size + j] *
                       kv_a_weight[weight_row * hidden_size + j];
            }
            k_pe[batch_idx * 1 * q_len * qk_rope_head_dim + seq_idx * qk_rope_head_dim + i] = sum;
        }
    }
    
    // Compute compressed_kv (shared across heads)
    for (int i = threadIdx.x; i < kv_lora_rank; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            sum += hidden_states[batch_idx * q_len * hidden_size + seq_idx * hidden_size + j] *
                   kv_a_weight[i * hidden_size + j];
        }
        compressed_kv[i] = sum;
    }
    
    __syncthreads();
    
    // Apply RMSNorm to compressed_kv
    if (threadIdx.x == 0) {
        float variance = 0.0f;
        for (int i = 0; i < kv_lora_rank; i++) {
            variance += compressed_kv[i] * compressed_kv[i];
        }
        variance /= kv_lora_rank;
        float rms = rsqrtf(variance + eps);
        
        for (int i = 0; i < kv_lora_rank; i++) {
            compressed_kv[i] = compressed_kv[i] * rms * kv_norm_weight[i];
        }
    }
    
    __syncthreads();
    
    // Expand with kv_b_proj for this head
    int total_qk_v_dim = qk_nope_head_dim + v_head_dim;
    
    // Compute k_nope
    for (int i = threadIdx.x; i < qk_nope_head_dim; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < kv_lora_rank; j++) {
            sum += compressed_kv[j] * kv_b_weight[head_idx * total_qk_v_dim * kv_lora_rank + i * kv_lora_rank + j];
        }
        k_nope[batch_idx * num_heads * q_len * qk_nope_head_dim + head_idx * q_len * qk_nope_head_dim + seq_idx * qk_nope_head_dim + i] = sum;
    }
    
    // Compute value_states
    for (int i = threadIdx.x; i < v_head_dim; i += blockDim.x) {
        float sum = 0.0f;
        int v_offset = qk_nope_head_dim;
        for (int j = 0; j < kv_lora_rank; j++) {
            sum += compressed_kv[j] * kv_b_weight[head_idx * total_qk_v_dim * kv_lora_rank + (v_offset + i) * kv_lora_rank + j];
        }
        value_states[batch_idx * num_heads * q_len * v_head_dim + head_idx * q_len * v_head_dim + seq_idx * v_head_dim + i] = sum;
    }
}

std::vector<torch::Tensor> fused_kv_proj_hip(
    torch::Tensor hidden_states,
    torch::Tensor kv_a_weight,
    torch::Tensor kv_b_weight,
    torch::Tensor kv_norm_weight,
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
    int shared_mem_size = kv_lora_rank * sizeof(float);
    
    fused_kv_proj_kernel<<<grid, 256, shared_mem_size>>>(
        hidden_states.data_ptr<float>(),
        kv_a_weight.data_ptr<float>(),
        kv_b_weight.data_ptr<float>(),
        kv_norm_weight.data_ptr<float>(),
        k_nope.data_ptr<float>(),
        k_pe.data_ptr<float>(),
        value_states.data_ptr<float>(),
        bsz, q_len, hidden_states.size(2),
        kv_lora_rank, qk_rope_head_dim, num_heads, qk_nope_head_dim, v_head_dim, eps
    );
    
    return {k_nope, k_pe, value_states};
}
"""

# Compile kernels
fused_q_proj = load_inline(
    name="fused_q_proj",
    cpp_sources=fused_q_proj_cpp_source,
    functions=["fused_q_proj_hip"],
    verbose=True,
)

fused_kv_proj = load_inline(
    name="fused_kv_proj",
    cpp_sources=fused_kv_proj_cpp_source,
    functions=["fused_kv_proj_hip"],
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

        # Original query projection (since fused version has bugs)
        compressed_q = self.q_a_layernorm(self.q_a_proj(hidden_states))
        query_states = self.q_b_proj(compressed_q)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)

        # Fused KV projection with compression
        kv_result = fused_kv_proj.fused_kv_proj_hip(
            hidden_states,
            self.kv_a_proj_with_mqa.weight,
            self.kv_b_proj.weight,
            self.kv_a_layernorm.weight,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.num_heads,
            self.qk_nope_head_dim,
            self.v_head_dim,
            self.kv_a_layernorm.variance_epsilon
        )
        
        k_nope = kv_result[0]
        k_pe = kv_result[1]
        value_states = kv_result[2]

        # Split query into nope and rope components
        q_nope, q_pe = torch.split(query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Rotary embeddings
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin)

        # Assemble full query and key states
        query_states = torch.empty(bsz, self.num_heads, q_len, self.q_head_dim,
                                   device=hidden_states.device, dtype=hidden_states.dtype)
        query_states[:, :, :, :self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim:] = q_pe

        key_states = torch.empty(bsz, self.num_heads, q_len, self.q_head_dim,
                                 device=hidden_states.device, dtype=hidden_states.dtype)
        key_states[:, :, :, :self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim:] = k_pe.expand(-1, self.num_heads, -1, -1)

        # Compute attention (standard PyTorch implementation)
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