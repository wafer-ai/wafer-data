import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Combined kernel source with optimizations
combined_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <float.h>

// ============ OPTIMIZED RMSNORM KERNEL ============
__global__ void rmsnorm_kernel_v2(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int hidden_size,
    float eps,
    int total_tokens
) {
    int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;
    
    const float* token_input = input + (long long)token_idx * hidden_size;
    float* token_output = output + (long long)token_idx * hidden_size;
    
    // Use float4 for vectorized loads
    float sum_sq = 0.0f;
    int vec_size = hidden_size / 4;
    const float4* input_vec = reinterpret_cast<const float4*>(token_input);
    
    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        float4 val = input_vec[i];
        sum_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }
    
    // Warp reduction using shuffle
    for (int offset = 32; offset > 0; offset /= 2) {
        sum_sq += __shfl_down(sum_sq, offset);
    }
    
    __shared__ float shared_sum[8];
    int lane = threadIdx.x % 64;
    int warp_id = threadIdx.x / 64;
    
    if (lane == 0) shared_sum[warp_id] = sum_sq;
    __syncthreads();
    
    if (threadIdx.x == 0) {
        float total = 0.0f;
        int nwarps = (blockDim.x + 63) / 64;
        for (int i = 0; i < nwarps; i++) total += shared_sum[i];
        shared_sum[0] = rsqrtf(total / hidden_size + eps);
    }
    __syncthreads();
    
    float rsqrt_var = shared_sum[0];
    
    // Vectorized write
    float4* output_vec = reinterpret_cast<float4*>(token_output);
    const float4* weight_vec = reinterpret_cast<const float4*>(weight);
    
    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        float4 val = input_vec[i];
        float4 w = weight_vec[i];
        float4 out;
        out.x = val.x * rsqrt_var * w.x;
        out.y = val.y * rsqrt_var * w.y;
        out.z = val.z * rsqrt_var * w.z;
        out.w = val.w * rsqrt_var * w.w;
        output_vec[i] = out;
    }
}

torch::Tensor rmsnorm_hip(torch::Tensor input, torch::Tensor weight, float eps) {
    auto output = torch::empty_like(input);
    int hidden_size = input.size(-1);
    int total_tokens = input.numel() / hidden_size;
    rmsnorm_kernel_v2<<<total_tokens, 256>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        output.data_ptr<float>(), hidden_size, eps, total_tokens);
    return output;
}

// ============ FUSED ROPE KERNEL ============
__global__ void fused_rope_kernel(
    const float* __restrict__ q_pe_in,
    const float* __restrict__ k_pe_in,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    float* __restrict__ q_pe_out,
    float* __restrict__ k_pe_out,
    int batch_size,
    int num_heads,
    int seq_len,
    int rope_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = rope_dim / 2;
    
    int total_q = batch_size * num_heads * seq_len * rope_dim;
    int total_k = batch_size * seq_len * rope_dim;
    
    // Process Q elements
    if (idx < total_q) {
        int d = idx % rope_dim;
        int s = (idx / rope_dim) % seq_len;
        int h = (idx / (rope_dim * seq_len)) % num_heads;
        int b = idx / (rope_dim * seq_len * num_heads);
        
        float cos_val = cos_cache[s * rope_dim + d];
        float sin_val = sin_cache[s * rope_dim + d];
        
        int partner_d = (d < half_dim) ? (d + half_dim) : (d - half_dim);
        int cur_idx = b * num_heads * seq_len * rope_dim + h * seq_len * rope_dim + s * rope_dim;
        
        float x = q_pe_in[cur_idx + d];
        float x_partner = q_pe_in[cur_idx + partner_d];
        float rotated = (d < half_dim) ? -x_partner : x_partner;
        
        q_pe_out[idx] = x * cos_val + rotated * sin_val;
    }
    
    // Process K elements
    if (idx < total_k) {
        int d = idx % rope_dim;
        int s = (idx / rope_dim) % seq_len;
        int b = idx / (rope_dim * seq_len);
        
        float cos_val = cos_cache[s * rope_dim + d];
        float sin_val = sin_cache[s * rope_dim + d];
        
        int partner_d = (d < half_dim) ? (d + half_dim) : (d - half_dim);
        int cur_idx = b * seq_len * rope_dim + s * rope_dim;
        
        float x = k_pe_in[cur_idx + d];
        float x_partner = k_pe_in[cur_idx + partner_d];
        float rotated = (d < half_dim) ? -x_partner : x_partner;
        
        k_pe_out[idx] = x * cos_val + rotated * sin_val;
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
    
    int total = std::max(batch_size * num_heads * seq_len * rope_dim, 
                         batch_size * seq_len * rope_dim);
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    
    fused_rope_kernel<<<num_blocks, block_size>>>(
        q_pe.data_ptr<float>(), k_pe.data_ptr<float>(),
        cos_cache.data_ptr<float>(), sin_cache.data_ptr<float>(),
        q_out.data_ptr<float>(), k_out.data_ptr<float>(),
        batch_size, num_heads, seq_len, rope_dim);
    
    return {q_out, k_out};
}

// ============ OPTIMIZED MASKED SOFTMAX KERNEL ============
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 32; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_xor(val, offset));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 32; offset > 0; offset /= 2)
        val += __shfl_xor(val, offset);
    return val;
}

__global__ void masked_softmax_kernel_v2(
    float* __restrict__ attn_weights,
    int seq_len,
    int num_heads,
    int batch_size
) {
    int batch_head = blockIdx.x;
    int row = blockIdx.y;
    
    int total_heads = batch_size * num_heads;
    if (batch_head >= total_heads || row >= seq_len) return;
    
    float* row_ptr = attn_weights + (long long)batch_head * seq_len * seq_len + (long long)row * seq_len;
    int valid_len = row + 1;
    
    // Step 1: Find max with vectorization where possible
    float max_val = -FLT_MAX;
    
    // Handle aligned portion with float4
    int vec_valid = valid_len / 4;
    const float4* row_vec = reinterpret_cast<const float4*>(row_ptr);
    
    for (int i = threadIdx.x; i < vec_valid; i += blockDim.x) {
        float4 v = row_vec[i];
        max_val = fmaxf(max_val, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    }
    
    // Handle remainder
    for (int j = vec_valid * 4 + threadIdx.x; j < valid_len; j += blockDim.x) {
        max_val = fmaxf(max_val, row_ptr[j]);
    }
    
    max_val = warp_reduce_max(max_val);
    
    __shared__ float shared_max[8];
    int warp_id = threadIdx.x / 64;
    int lane = threadIdx.x % 64;
    if (lane == 0) shared_max[warp_id] = max_val;
    __syncthreads();
    
    if (threadIdx.x == 0) {
        float m = -FLT_MAX;
        int nwarps = (blockDim.x + 63) / 64;
        for (int i = 0; i < nwarps; i++) m = fmaxf(m, shared_max[i]);
        shared_max[0] = m;
    }
    __syncthreads();
    max_val = shared_max[0];
    
    // Step 2: Compute exp and sum
    float sum = 0.0f;
    float4* row_vec_rw = reinterpret_cast<float4*>(row_ptr);
    
    for (int i = threadIdx.x; i < vec_valid; i += blockDim.x) {
        float4 v = row_vec[i];
        float4 e;
        e.x = expf(v.x - max_val);
        e.y = expf(v.y - max_val);
        e.z = expf(v.z - max_val);
        e.w = expf(v.w - max_val);
        row_vec_rw[i] = e;
        sum += e.x + e.y + e.z + e.w;
    }
    
    for (int j = vec_valid * 4 + threadIdx.x; j < valid_len; j += blockDim.x) {
        float exp_val = expf(row_ptr[j] - max_val);
        row_ptr[j] = exp_val;
        sum += exp_val;
    }
    
    // Set masked positions to 0
    int vec_full = seq_len / 4;
    for (int i = (valid_len + 3) / 4 + threadIdx.x; i < vec_full; i += blockDim.x) {
        row_vec_rw[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    for (int j = valid_len + threadIdx.x; j < ((valid_len + 3) / 4) * 4 && j < seq_len; j += blockDim.x) {
        row_ptr[j] = 0.0f;
    }
    
    sum = warp_reduce_sum(sum);
    __shared__ float shared_sum[8];
    if (lane == 0) shared_sum[warp_id] = sum;
    __syncthreads();
    
    if (threadIdx.x == 0) {
        float s = 0.0f;
        int nwarps = (blockDim.x + 63) / 64;
        for (int i = 0; i < nwarps; i++) s += shared_sum[i];
        shared_sum[0] = 1.0f / s;
    }
    __syncthreads();
    float inv_sum = shared_sum[0];
    
    // Step 3: Normalize
    for (int i = threadIdx.x; i < vec_valid; i += blockDim.x) {
        float4 v = row_vec_rw[i];
        v.x *= inv_sum;
        v.y *= inv_sum;
        v.z *= inv_sum;
        v.w *= inv_sum;
        row_vec_rw[i] = v;
    }
    
    for (int j = vec_valid * 4 + threadIdx.x; j < valid_len; j += blockDim.x) {
        row_ptr[j] *= inv_sum;
    }
}

void masked_softmax_hip(torch::Tensor attn_weights, int seq_len, int num_heads, int batch_size) {
    dim3 blocks(batch_size * num_heads, seq_len);
    masked_softmax_kernel_v2<<<blocks, 256>>>(
        attn_weights.data_ptr<float>(), seq_len, num_heads, batch_size);
}
"""

cpp_decl = """
torch::Tensor rmsnorm_hip(torch::Tensor input, torch::Tensor weight, float eps);
std::vector<torch::Tensor> fused_rope_hip(torch::Tensor q_pe, torch::Tensor k_pe, torch::Tensor cos_cache, torch::Tensor sin_cache);
void masked_softmax_hip(torch::Tensor attn_weights, int seq_len, int num_heads, int batch_size);
"""

combined_module = load_inline(
    name="combined_hip_v8",
    cpp_sources=cpp_decl,
    cuda_sources=combined_source,
    functions=["rmsnorm_hip", "fused_rope_hip", "masked_softmax_hip"],
    verbose=False,
    extra_cuda_cflags=["-O3"]
)


class DeepSeekRMSNormOptimized(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return combined_module.rmsnorm_hip(hidden_states.contiguous(), self.weight, self.variance_epsilon)


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

        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_a_layernorm = DeepSeekRMSNormOptimized(q_lora_rank)
        self.q_b_proj = nn.Linear(q_lora_rank, num_attention_heads * self.q_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = DeepSeekRMSNormOptimized(kv_lora_rank)
        self.kv_b_proj = nn.Linear(kv_lora_rank, num_attention_heads * (qk_nope_head_dim + v_head_dim), bias=False)

        self.o_proj = nn.Linear(num_attention_heads * v_head_dim, hidden_size, bias=False)
        self.rotary_emb = DeepSeekRotaryEmbedding(qk_rope_head_dim, max_position_embeddings=max_position_embeddings, base=rope_theta)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Query projection
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # KV projection
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Apply RoPE
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        q_pe_rope, k_pe_rope = combined_module.fused_rope_hip(
            q_pe.contiguous(), k_pe.squeeze(1).contiguous(),
            cos.contiguous(), sin.contiguous())
        k_pe_rope = k_pe_rope.unsqueeze(1)

        # Assemble Q and K using cat (efficient)
        query_states = torch.cat([q_nope, q_pe_rope], dim=-1)
        key_states = torch.cat([k_nope, k_pe_rope.expand(-1, self.num_heads, -1, -1)], dim=-1)

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        combined_module.masked_softmax_hip(attn_weights, q_len, self.num_heads, bsz)

        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output


def custom_kernel(inputs):
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
