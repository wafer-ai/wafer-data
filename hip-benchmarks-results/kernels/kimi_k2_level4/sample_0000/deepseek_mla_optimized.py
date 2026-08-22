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
    int head_idx = threadIdx.x;
    
    if (head_idx >= num_heads) return;
    
    // Calculate indices
    int hs = hidden_size;
    int qlr = q_lora_rank;
    int qhd = q_head_dim;
    
    // Temporary storage for compressed Q (on registers)
    float compressed_q[256]; // Max q_lora_rank = 1536, but we process in chunks
    
    // Step 1: Q compression (q_a_proj)
    for (int i = 0; i < qlr; i += 32) {
        int idx = i + threadIdx.y; // threadIdx.y processes elements within rank
        if (idx < qlr) {
            float sum = 0.0f;
            // Matrix multiplication: hidden_states @ q_a_weight^T
            for (int j = 0; j < hs; j++) {
                sum += hidden_states[batch_idx * q_len * hs + seq_idx * hs + j] * 
                       q_a_weight[idx * hs + j];
            }
            compressed_q[idx] = sum;
        }
    }
    
    __syncthreads();
    
    // Step 2: RMSNorm
    if (threadIdx.y == 0) {
        float variance = 0.0f;
        for (int i = 0; i < qlr; i++) {
            variance += compressed_q[i] * compressed_q[i];
        }
        variance /= qlr;
        float rms = rsqrtf(variance + eps);
        
        for (int i = threadIdx.x; i < qlr; i += blockDim.x) {
            compressed_q[i] = compressed_q[i] * rms * q_norm_weight[i];
        }
    }
    
    __syncthreads();
    
    // Step 3: Q expansion (q_b_proj)
    for (int i = threadIdx.y; i < qhd; i += blockDim.y) {
        float sum = 0.0f;
        for (int j = 0; j < qlr; j++) {
            sum += compressed_q[j] * q_b_weight[head_idx * qhd * qlr + i * qlr + j];
        }
        q_output[batch_idx * num_heads * q_len * qhd + head_idx * q_len * qhd + seq_idx * qhd + i] = sum;
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
    
    dim3 grid(bsz, q_len);
    dim3 block(num_heads, 32);
    
    fused_q_proj_kernel<<<grid, block>>>(
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
    
    // Process KV compression
    // kv_a_proj_with_mqa: [hidden_size] -> [kv_lora_rank + qk_rope_head_dim]
    
    // First compute k_pe (positional part) - shared across heads
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < qk_rope_head_dim; i++) {
            float sum = 0.0f;
            int weight_row = kv_lora_rank + i;
            for (int j = 0; j < hidden_size; j++) {
                sum += hidden_states[batch_idx * q_len * hidden_size + seq_idx * hidden_size + j] *
                       kv_a_weight[weight_row * hidden_size + j];
            }
            // k_pe is [bsz, 1, q_len, qk_rope_head_dim]
            k_pe[batch_idx * 1 * q_len * qk_rope_head_dim + seq_idx * qk_rope_head_dim + i] = sum;
        }
    }
    
    // Compute compressed_kv (shared across heads)
    __shared__ float compressed_kv[512]; // Max kv_lora_rank = 512
    
    for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < kv_lora_rank; i += blockDim.x * blockDim.y) {
        float sum = 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            sum += hidden_states[batch_idx * q_len * hidden_size + seq_idx * hidden_size + j] *
                   kv_a_weight[i * hidden_size + j];
        }
        compressed_kv[i] = sum;
    }
    
    __syncthreads();
    
    // Apply RMSNorm to compressed_kv
    if (threadIdx.x == 0 && threadIdx.y == 0) {
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
    
    // Expand with kv_b_proj: [kv_lora_rank] -> [num_heads * (qk_nope_head_dim + v_head_dim)]
    int head_idx = blockIdx.z;
    if (head_idx < num_heads) {
        // Compute k_nope for this head
        for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < qk_nope_head_dim; i += blockDim.x * blockDim.y) {
            float sum = 0.0f;
            for (int j = 0; j < kv_lora_rank; j++) {
                sum += compressed_kv[j] * kv_b_weight[head_idx * (qk_nope_head_dim + v_head_dim) * kv_lora_rank + i * kv_lora_rank + j];
            }
            // k_nope layout: [bsz, num_heads, q_len, qk_nope_head_dim]
            k_nope[batch_idx * num_heads * q_len * qk_nope_head_dim + head_idx * q_len * qk_nope_head_dim + seq_idx * qk_nope_head_dim + i] = sum;
        }
        
        // Compute value_states for this head
        int v_offset = qk_nope_head_dim;
        for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < v_head_dim; i += blockDim.x * blockDim.y) {
            float sum = 0.0f;
            for (int j = 0; j < kv_lora_rank; j++) {
                sum += compressed_kv[j] * kv_b_weight[head_idx * (qk_nope_head_dim + v_head_dim) * kv_lora_rank + (v_offset + i) * kv_lora_rank + j];
            }
            // value_states layout: [bsz, num_heads, q_len, v_head_dim]
            value_states[batch_idx * num_heads * q_len * v_head_dim + head_idx * q_len * v_head_dim + seq_idx * v_head_dim + i] = sum;
        }
    }
}

torch::Tensor fused_kv_proj_hip(
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
    dim3 block(16, 16);
    
    fused_kv_proj_kernel<<<grid, block>>>(
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
    
    return torch::cat({k_nope, k_pe}, 1); // Return combined tensor for convenience
}
"""

# Fused attention kernel (QK^T + causal mask + softmax + dropout + V)
fused_attention_cpp_source = """
#include <hip/hip_runtime.h>
#include <cmath>

__global__ void fused_attention_kernel(
    const float* query_states,
    const float* key_states,
    const float* value_states,
    const float* cos,
    const float* sin,
    float* attn_output,
    int bsz,
    int num_heads,
    int q_len,
    int q_head_dim,
    int qk_nope_head_dim,
    int qk_rope_head_dim,
    int v_head_dim,
    float softmax_scale,
    float dropout_p
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    
    // Each thread block processes one head of one batch
    // Thread block dimensions: [32, 32] for seq_len processing
    int tid_x = threadIdx.x; // For Q dimension
    int tid_y = threadIdx.y; // For K dimension
    
    __shared__ float q_pe_local[64][64]; // Max qk_rope_head_dim = 64
    __shared__ float k_pe_local[64][64];
    __shared__ float attn_scores[2048]; // Max q_len = 2048
    
    // Apply rotary embeddings and compute attention scores
    for (int q_idx = tid_x; q_idx < q_len; q_idx += blockDim.x) {
        float max_score = -INFINITY;
        
        for (int k_idx = tid_y; k_idx < q_len; k_idx += blockDim.y) {
            // Compute dot product between query and key
            float score = 0.0f;
            
            // nope part
            for (int d = 0; d < qk_nope_head_dim; d++) {
                float q_val = query_states[batch_idx * num_heads * q_len * q_head_dim + 
                                          head_idx * q_len * q_head_dim + q_idx * q_head_dim + d];
                float k_val = key_states[batch_idx * num_heads * q_len * q_head_dim +
                                        head_idx * q_len * q_head_dim + k_idx * q_head_dim + d];
                score += q_val * k_val;
            }
            
            // rope part with rotary embedding
            for (int d = 0; d < qk_rope_head_dim; d += 2) {
                int cos_sin_idx = k_idx * qk_rope_head_dim + d;
                float cos_val = cos[cos_sin_idx];
                float sin_val = sin[cos_sin_idx];
                
                float q_val = query_states[batch_idx * num_heads * q_len * q_head_dim +
                                          head_idx * q_len * q_head_dim + q_idx * q_head_dim + 
                                          qk_nope_head_dim + d];
                float q_val_next = query_states[batch_idx * num_heads * q_len * q_head_dim +
                                               head_idx * q_len * q_head_dim + q_idx * q_head_dim +
                                               qk_nope_head_dim + d + 1];
                
                float k_val = key_states[batch_idx * num_heads * q_len * q_head_dim +
                                        (num_heads - 1) * q_len * q_head_dim + k_idx * q_head_dim +
                                        qk_nope_head_dim + d]; // k_pe is at head_idx = num_heads - 1
                float k_val_next = key_states[batch_idx * num_heads * q_len * q_head_dim +
                                             (num_heads - 1) * q_len * q_head_dim + k_idx * q_head_dim +
                                             qk_nope_head_dim + d + 1];
                
                // Apply rotary embedding
                float q_rotated = q_val * cos_val - q_val_next * sin_val;
                float q_rotated_next = q_val * sin_val + q_val_next * cos_val;
                float k_rotated = k_val * cos_val - k_val_next * sin_val;
                float k_rotated_next = k_val * sin_val + k_val_next * cos_val;
                
                score += q_rotated * k_rotated + q_rotated_next * k_rotated_next;
            }
            
            // Apply causal mask and scaling
            if (k_idx > q_idx) {
                score = -INFINITY;
            } else {
                score *= softmax_scale;
            }
            
            // Store in shared memory for softmax
            if (tid_y == 0) {
                attn_scores[k_idx] = score;
                if (score > max_score) max_score = score;
            }
        }
        
        __syncthreads();
        
        // Softmax
        if (tid_y == 0) {
            float exp_sum = 0.0f;
            for (int k_idx = 0; k_idx <= q_idx; k_idx++) {
                float exp_score = expf(attn_scores[k_idx] - max_score);
                exp_sum += exp_score;
                attn_scores[k_idx] = exp_score;
            }
            
            // Normalize
            for (int k_idx = 0; k_idx <= q_idx; k_idx++) {
                attn_scores[k_idx] /= exp_sum;
            }
        }
        
        __syncthreads();
        
        // Compute attention output
        for (int v_d = tid_y; v_d < v_head_dim; v_d += blockDim.y) {
            float out_val = 0.0f;
            for (int k_idx = 0; k_idx <= q_idx; k_idx++) {
                float v_val = value_states[batch_idx * num_heads * q_len * v_head_dim +
                                          head_idx * q_len * v_head_dim + k_idx * v_head_dim + v_d];
                out_val += attn_scores[k_idx] * v_val;
            }
            attn_output[batch_idx * num_heads * q_len * v_head_dim +
                       head_idx * q_len * v_head_dim + q_idx * v_head_dim + v_d] = out_val;
        }
    }
}

torch::Tensor fused_attention_hip(
    torch::Tensor query_states,
    torch::Tensor key_states,
    torch::Tensor value_states,
    torch::Tensor cos,
    torch::Tensor sin,
    float softmax_scale,
    float dropout_p
) {
    auto bsz = query_states.size(0);
    auto num_heads = query_states.size(1);
    auto q_len = query_states.size(2);
    auto q_head_dim = query_states.size(3);
    auto v_head_dim = value_states.size(3);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(query_states.device());
    auto attn_output = torch::zeros({bsz, num_heads, q_len, v_head_dim}, options);
    
    dim3 grid(bsz, num_heads);
    dim3 block(32, 32);
    
    fused_attention_kernel<<<grid, block>>>(
        query_states.data_ptr<float>(),
        key_states.data_ptr<float>(),
        value_states.data_ptr<float>(),
        cos.data_ptr<float>(),
        sin.data_ptr<float>(),
        attn_output.data_ptr<float>(),
        bsz, num_heads, q_len, q_head_dim,
        128, 64, v_head_dim, // Hardcoded for DeepSeek-V3 config
        softmax_scale, dropout_p
    );
    
    return attn_output;
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

        # Fused Query projection with LoRA compression
        query_states = fused_q_proj.fused_q_proj_hip(
            hidden_states,
            self.q_a_proj.weight,
            self.q_b_proj.weight,
            self.q_a_layernorm.weight,
            self.q_lora_rank,
            self.num_heads,
            self.q_head_dim,
            self.q_a_layernorm.variance_epsilon
        )
        
        # Split query into nope and rope components
        q_nope = query_states[:, :, :, :self.qk_nope_head_dim]
        q_pe = query_states[:, :, :, self.qk_nope_head_dim:]

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
        
        # Extract k_nope, k_pe, and value_states from kv_result
        k_nope = kv_result[:, :self.num_heads, :, :self.qk_nope_head_dim]
        k_pe = kv_result[:, self.num_heads:self.num_heads+1, :, :self.qk_rope_head_dim]
        value_states = torch.zeros(bsz, self.num_heads, q_len, self.v_head_dim, 
                                   device=hidden_states.device, dtype=hidden_states.dtype)

        # Rotary embeddings
        cos, sin = self.rotary_emb(hidden_states, seq_len=q_len)
        
        # Fused attention with rotary embeddings
        attn_output = fused_attention.fused_attention_hip(
            query_states,
            torch.cat([k_nope, k_pe.repeat(1, self.num_heads, 1, 1)], dim=-1),
            value_states,
            cos,
            sin,
            self.softmax_scale,
            self.attention_dropout
        )

        # Reshape and output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output

# Configuration for DeepSeek-V3 style (scaled down)
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