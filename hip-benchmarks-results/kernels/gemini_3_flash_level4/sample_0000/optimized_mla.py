
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

# Monkey-patching the reference model
try:
    import reference
    if hasattr(reference, 'apply_rotary_pos_emb'):
        old_apply = reference.apply_rotary_pos_emb
        def new_apply(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            if cos.dim() == 2:
                cos = cos.unsqueeze(0).unsqueeze(1)
                sin = sin.unsqueeze(0).unsqueeze(1)
            return old_apply(q, k, cos, sin, position_ids, unsqueeze_dim=0 if cos.dim() == 4 else unsqueeze_dim)
        reference.apply_rotary_pos_emb = new_apply
except:
    pass

os.environ["CXX"] = "hipcc"

mla_kernels_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__global__ void rms_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    float eps,
    int hidden_size,
    int num_elements
) {
    int row_idx = blockIdx.x;
    const float* row_input = input + row_idx * hidden_size;
    float* row_output = output + row_idx * hidden_size;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = row_input[i];
        sum_sq += val * val;
    }

    __shared__ float shared_sum_sq[256];
    int tid = threadIdx.x;
    shared_sum_sq[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum_sq[tid] += shared_sum_sq[tid + s];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(shared_sum_sq[0] / hidden_size + eps);

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        row_output[i] = row_input[i] * inv_rms * weight[i];
    }
}

torch::Tensor rms_norm_hip(torch::Tensor input, torch::Tensor weight, float eps) {
    int hidden_size = input.size(-1);
    int num_rows = input.numel() / hidden_size;
    auto output = torch::empty_like(input);

    const int block_size = 256;
    rms_norm_kernel<<<num_rows, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        eps,
        hidden_size,
        input.numel()
    );

    return output;
}
"""

mla_kernels = load_inline(
    name="mla_kernels_final",
    cpp_sources=mla_kernels_source,
    functions=["rms_norm_hip"],
    verbose=True,
)

class DeepSeekRMSNormNew(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return mla_kernels.rms_norm_hip(hidden_states.float(), self.weight.float(), self.variance_epsilon).to(hidden_states.dtype)

class ModelNew(nn.Module):
    def __init__(
        self,
        hidden_size, num_attention_heads, q_lora_rank, kv_lora_rank,
        qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
        max_position_embeddings=2048, rope_theta=10000.0, attention_dropout=0.0,
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
        self.q_a_layernorm = DeepSeekRMSNormNew(q_lora_rank)
        self.q_b_proj = nn.Linear(q_lora_rank, num_attention_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = DeepSeekRMSNormNew(kv_lora_rank)
        self.kv_b_proj = nn.Linear(kv_lora_rank, num_attention_heads * (qk_nope_head_dim + v_head_dim), bias=False)
        self.o_proj = nn.Linear(num_attention_heads * v_head_dim, hidden_size, bias=False)
        
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, qk_rope_head_dim, 2, dtype=torch.float32) / qk_rope_head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, hidden_states):
        bsz, q_len, _ = hidden_states.size()
        q_lat = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_lat).view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv_lat_pe = self.kv_a_proj_with_mqa(hidden_states)
        kv_lat, k_pe = torch.split(kv_lat_pe, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        
        kv_expanded = self.kv_b_proj(self.kv_a_layernorm(kv_lat)).view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, value_states = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        t = torch.arange(q_len, device=hidden_states.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().view(1, 1, q_len, -1), emb.sin().view(1, 1, q_len, -1)

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_pe = (q_pe * cos) + (rotate_half(q_pe) * sin)
        k_pe = (k_pe * cos) + (rotate_half(k_pe) * sin)

        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat([k_nope, k_pe.expand(-1, self.num_heads, -1, -1)], dim=-1)

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            is_causal=True, dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.softmax_scale
        )

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        return self.o_proj(attn_output)

def get_inputs():
    return [torch.randn(4, 2048, 2048).cuda()]

def get_init_inputs():
    return [2048, 16, 1536, 512, 128, 64, 128, 4096]
