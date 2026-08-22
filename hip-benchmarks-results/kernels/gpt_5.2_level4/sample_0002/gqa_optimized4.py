import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ.setdefault("CXX", "hipcc")

# Custom HIP kernel: in-place RoPE for Q/K (FP32)
_rope_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

__global__ void rope_inplace_f32(
    float* __restrict__ x,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    int B, int H, int L, int D,
    int64_t xs0, int64_t xs1, int64_t xs2,
    int64_t cs2, int64_t cs3,
    int64_t ss2, int64_t ss3
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int half = D >> 1;
    const int total = B * H * L * half;
    if (idx >= total) return;

    int t = idx;
    const int d = t % half; t /= half;
    const int l = t % L;    t /= L;
    const int h = t % H;    t /= H;
    const int b = t;

    float* base = x + b*xs0 + h*xs1 + l*xs2;

    const int d0 = d;
    const int d1 = d + half;

    const float x0 = base[d0];
    const float x1 = base[d1];

    const float c0 = cos[l*cs2 + d0*cs3];
    const float s0 = sin[l*ss2 + d0*ss3];
    const float c1 = cos[l*cs2 + d1*cs3];
    const float s1 = sin[l*ss2 + d1*ss3];

    const float x0_rot = -x1;
    const float x1_rot =  x0;

    base[d0] = x0 * c0 + x0_rot * s0;
    base[d1] = x1 * c1 + x1_rot * s1;
}

torch::Tensor rope_inplace(torch::Tensor x, torch::Tensor cos, torch::Tensor sin) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA/HIP");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "FP32 only");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");

    const int B = (int)x.size(0);
    const int H = (int)x.size(1);
    const int L = (int)x.size(2);
    const int D = (int)x.size(3);
    TORCH_CHECK((D % 2) == 0, "D must be even");

    const int64_t xs0 = x.stride(0);
    const int64_t xs1 = x.stride(1);
    const int64_t xs2 = x.stride(2);

    const int64_t cs2 = cos.stride(2);
    const int64_t cs3 = cos.stride(3);
    const int64_t ss2 = sin.stride(2);
    const int64_t ss3 = sin.stride(3);

    const int half = D >> 1;
    const int total = B * H * L * half;

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    hipStream_t stream = (hipStream_t)at::cuda::getDefaultCUDAStream();
    hipLaunchKernelGGL(
        rope_inplace_f32,
        dim3(blocks),
        dim3(threads),
        0,
        stream,
        (float*)x.data_ptr<float>(),
        (const float*)cos.data_ptr<float>(),
        (const float*)sin.data_ptr<float>(),
        B, H, L, D,
        xs0, xs1, xs2,
        cs2, cs3,
        ss2, ss3
    );
    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rope_inplace", &rope_inplace, "RoPE in-place (FP32)");
}
"""

_rope_ext = load_inline(
    name="rope_inplace_ext_v3",
    cpp_sources="",
    cuda_sources=_rope_src,
    functions=None,
    extra_cuda_cflags=["-O3", "-ffast-math"],
    with_cuda=True,
    verbose=False,
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

        # Keep original modules so initialization matches the reference.
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

        # Packed QKV weight for single GEMM (copied from separately initialized weights)
        total_out = (num_attention_heads + 2 * num_key_value_heads) * head_dim
        w_qkv = torch.empty(total_out, hidden_size, dtype=torch.float32)
        q_end = num_attention_heads * head_dim
        k_end = q_end + num_key_value_heads * head_dim
        w_qkv[:q_end].copy_(self.q_proj.weight.detach())
        w_qkv[q_end:k_end].copy_(self.k_proj.weight.detach())
        w_qkv[k_end:].copy_(self.v_proj.weight.detach())
        self.register_buffer("w_qkv", w_qkv, persistent=False)
        self.q_end = q_end
        self.k_end = k_end

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(dtype=torch.float32)
        bsz, q_len, _ = hidden_states.size()

        # Fused QKV projection (single GEMM)
        qkv = F.linear(hidden_states, self.w_qkv)
        q = qkv[..., : self.q_end]
        k = qkv[..., self.q_end : self.k_end]
        v = qkv[..., self.k_end :]

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(v, seq_len=q_len)

        # Custom RoPE
        _rope_ext.rope_inplace(q, cos, sin)
        _rope_ext.rope_inplace(k, cos, sin)

        # SDPA with causal + GQA support
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=(self.attention_dropout if self.training else 0.0),
            is_causal=True,
            scale=(self.head_dim ** (-0.5)),
            enable_gqa=True,
        )

        # Avoid explicit contiguous if possible (Linear can often handle strided inputs)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_out)


# Prompt configuration
batch_size = 4
seq_len = 2048
hidden_size = 4096
num_attention_heads = 32
num_key_value_heads = 8
head_dim = 128
max_position_embeddings = 4096


def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float32)]


def get_init_inputs():
    return [hidden_size, num_attention_heads, num_key_value_heads, head_dim, max_position_embeddings]
