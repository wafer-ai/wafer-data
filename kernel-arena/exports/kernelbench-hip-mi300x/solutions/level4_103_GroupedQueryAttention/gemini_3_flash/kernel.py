
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

rope_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void rope_reshape_kernel(
    const float* __restrict__ qkv_in,   // (bsz, q_len, total_heads * head_dim)
    const float* __restrict__ cos_table, // (q_len, head_dim)
    const float* __restrict__ sin_table, // (q_len, head_dim)
    float* __restrict__ q_out,           // (bsz, n_heads, q_len, head_dim)
    float* __restrict__ k_out,           // (bsz, n_kv_heads, q_len, head_dim)
    float* __restrict__ v_out,           // (bsz, n_kv_heads, q_len, head_dim)
    int bsz, int q_len, int n_heads, int n_kv_heads, int head_dim
) {
    int b = blockIdx.z;
    int q = blockIdx.y;
    int h = blockIdx.x;
    int d = threadIdx.x;

    int total_heads = n_heads + 2 * n_kv_heads;
    int in_idx = ((b * q_len + q) * total_heads + h) * head_dim + d;
    float val = qkv_in[in_idx];

    if (h < n_heads + n_kv_heads) { // Q or K
        int rope_other_d = (d < head_dim / 2) ? d + head_dim / 2 : d - head_dim / 2;
        float val_other = qkv_in[((b * q_len + q) * total_heads + h) * head_dim + rope_other_d];
        float cos_val = cos_table[q * head_dim + d];
        float sin_val = sin_table[q * head_dim + d];
        
        float res;
        if (d < head_dim / 2) {
            res = val * cos_val - val_other * sin_val;
        } else {
            res = val * cos_val + val_other * sin_val;
        }

        if (h < n_heads) { // Q
            int out_idx = ((b * n_heads + h) * q_len + q) * head_dim + d;
            q_out[out_idx] = res;
        } else { // K
            int out_idx = ((b * n_kv_heads + (h - n_heads)) * q_len + q) * head_dim + d;
            k_out[out_idx] = res;
        }
    } else { // V
        int out_idx = ((b * n_kv_heads + (h - n_heads - n_kv_heads)) * q_len + q) * head_dim + d;
        v_out[out_idx] = val;
    }
}

std::vector<torch::Tensor> rope_reshape_hip(
    torch::Tensor qkv_in, torch::Tensor cos, torch::Tensor sin,
    int n_heads, int n_kv_heads, int head_dim
) {
    int bsz = qkv_in.size(0);
    int q_len = qkv_in.size(1);
    
    auto q_out = torch::empty({bsz, n_heads, q_len, head_dim}, qkv_in.options());
    auto k_out = torch::empty({bsz, n_kv_heads, q_len, head_dim}, qkv_in.options());
    auto v_out = torch::empty({bsz, n_kv_heads, q_len, head_dim}, qkv_in.options());

    dim3 grid(n_heads + 2 * n_kv_heads, q_len, bsz);
    dim3 block(head_dim);

    rope_reshape_kernel<<<grid, block>>>(
        qkv_in.data_ptr<float>(), cos.data_ptr<float>(), sin.data_ptr<float>(),
        q_out.data_ptr<float>(), k_out.data_ptr<float>(), v_out.data_ptr<float>(),
        bsz, q_len, n_heads, n_kv_heads, head_dim
    );

    return {q_out, k_out, v_out};
}
"""

rope_reshape = load_inline(
    name="rope_reshape",
    cpp_sources=rope_kernel_source,
    functions=["rope_reshape_hip"],
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

        self.qkv_proj = nn.Linear(hidden_size, (num_attention_heads + 2 * num_key_value_heads) * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, seq_len, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        qkv_output = self.qkv_proj(hidden_states)

        cos, sin = self.rotary_emb(qkv_output, seq_len=q_len)
        cos = cos.squeeze(0).squeeze(0)
        sin = sin.squeeze(0).squeeze(0)

        query_states, key_states, value_states = rope_reshape.rope_reshape_hip(
            qkv_output, cos, sin, self.num_heads, self.num_kv_heads, self.head_dim
        )

        key_states = self.repeat_kv(key_states, self.num_key_value_groups)
        value_states = self.repeat_kv(value_states, self.num_key_value_groups)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output
