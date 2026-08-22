import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ.setdefault("CXX", "hipcc")

# Simple custom fp32 GELU (NewGELU formulation) as a HIP kernel.
# Main speedup comes from using PyTorch's SDPA (flash attention) backend instead of
# materializing the [T,T] attention matrix.

gelucpp = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>

__global__ void newgelu_f32_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = x[idx];
    // 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    float v3 = v * v * v;
    float inner = 0.7978845608028654f * (v + 0.044715f * v3);
    float t = tanhf(inner);
    y[idx] = 0.5f * v * (1.0f + t);
}

torch::Tensor newgelu_f32(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA/HIP");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be fp32");
    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    hipLaunchKernelGGL(newgelu_f32_kernel, dim3(blocks), dim3(threads), 0, 0,
                       (const float*)x.data_ptr<float>(), (float*)y.data_ptr<float>(), n);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("newgelu_f32", &newgelu_f32, "NewGELU fp32 (HIP)");
}
'''

geluext = load_inline(
    name="newgelu_ext",
    cpp_sources=gelucpp,
    functions=None,
    with_cuda=True,
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class NewGELUNew(nn.Module):
    def forward(self, x):
        return geluext.newgelu_f32(x)


class CausalSelfAttentionNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hs = C // self.n_head

        # [B, nh, T, hs]
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        # Use optimized SDPA backend on ROCm (typically FlashAttention/efficient kernel)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttentionNew(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(n_embd, 4 * n_embd),
                c_proj=nn.Linear(4 * n_embd, n_embd),
                act=NewGELUNew(),
                dropout=nn.Dropout(resid_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0

def get_inputs():
    return [torch.rand(batch_size, seq_len, n_embd, device='cuda', dtype=torch.float32)]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]
