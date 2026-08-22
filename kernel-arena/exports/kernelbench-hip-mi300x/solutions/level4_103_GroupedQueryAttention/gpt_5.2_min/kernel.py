import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ.setdefault("CXX", "hipcc")

hip_src = r'''
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>

__global__ void repeat_kv_fp32_kernel(
    const float* __restrict__ inp,
    float* __restrict__ out,
    int B, int HKV, int S, int D, int groups
) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    long long total = (long long)B * (long long)HKV * (long long)groups * (long long)S * (long long)D;
    if ((long long)idx >= total) return;

    int d = idx % D;
    long long t = idx / D;
    int s = (int)(t % S);
    t /= S;
    int g = (int)(t % groups);
    t /= groups;
    int kv_h = (int)(t % HKV);
    int b = (int)(t / HKV);

    int h = kv_h * groups + g;

    long long out_off = (((long long)b * (HKV * groups) + h) * S + s) * D + d;
    long long in_off  = (((long long)b * HKV + kv_h) * S + s) * D + d;
    out[out_off] = inp[in_off];
}

torch::Tensor repeat_kv_fp32(torch::Tensor x, int64_t groups) {
    TORCH_CHECK(x.is_cuda(), "x must be on GPU");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "FP32 only");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "expected [B,HKV,S,D]");

    int B = (int)x.size(0);
    int HKV = (int)x.size(1);
    int S = (int)x.size(2);
    int D = (int)x.size(3);
    int H = HKV * (int)groups;

    auto out = torch::empty({B, H, S, D}, x.options());

    long long total = (long long)B * (long long)H * (long long)S * (long long)D;
    int block = 256;
    int grid = (int)((total + block - 1) / block);

    hipStream_t stream = at::hip::getDefaultHIPStream();
    hipLaunchKernelGGL(repeat_kv_fp32_kernel, dim3(grid), dim3(block), 0, stream,
        (const float*)x.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        B, HKV, S, D, (int)groups
    );
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("repeat_kv_fp32", &repeat_kv_fp32, "repeat_kv FP32 (contiguous)");
}
'''

ext = load_inline(
    name="gqa_repeat_ext",
    cpp_sources="",
    cuda_sources=hip_src,
    functions=None,
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


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

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(head_dim, max_position_embeddings=max_position_embeddings, base=rope_theta)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dtype != torch.float32:
            hidden_states = hidden_states.float()

        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        cos, sin = self.rotary_emb(v, seq_len=q_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k_rep = ext.repeat_kv_fp32(k.contiguous(), self.num_key_value_groups)
        v_rep = ext.repeat_kv_fp32(v.contiguous(), self.num_key_value_groups)

        # Let SDPA apply its own scaling (1/sqrt(head_dim)) to match reference.
        attn_out = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=None, dropout_p=0.0, is_causal=True)

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_out)


batch_size = 4
seq_len = 2048
hidden_size = 4096
num_attention_heads = 32
num_key_value_heads = 8
head_dim = 128
max_position_embeddings = 4096


def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float32)]


def get_init_inputs():
    return [hidden_size, num_attention_heads, num_key_value_heads, head_dim, max_position_embeddings]
