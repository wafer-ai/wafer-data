import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

os.environ["CXX"] = "hipcc"

# Correct custom GELU that matches the reference exactly
gelu_cpp_source = """
#include <hip/hip_runtime.h>
#include <math.h>

#define GELU_SCALING 0.044715f
#define SQRT_2_OVER_PI 0.7978845608028654f

__global__ void gelu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        float cube = xi * xi * xi;
        float inner = GELU_SCALING * cube + xi;
        float mult = sqrtf(2.0f / M_PI) * inner;
        float tanh_val = tanhf(mult);
        out[idx] = 0.5f * xi * (1.0f + tanh_val);
    }
}

torch::Tensor gelu_hip(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}
"""

gelu_custom = load_inline(
    name="gelu_op",
    cpp_sources=gelu_cpp_source,
    functions=["gelu_hip"],
    verbose=True,
)

class NewGELU(nn.Module):
    def __init__(self):
        super(NewGELU, self).__init__()
        self.gelu = gelu_custom
    
    def forward(self, x):
        return self.gelu.gelu_hip(x)

# Use PyTorch's optimized attention implementation
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        
        # Use PyTorch's built-in MHA which has optimized kernels
        self.mha = nn.MultiheadAttention(n_embd, n_head, dropout=attn_pdrop, batch_first=True)
        
        # For projection - need to split this into qkv for MHA
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))

    def forward(self, x):
        B, T, C = x.size()
        
        # Extract Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for MHA
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Apply PyTorch's optimized multi-head attention
        causal_mask = self.bias[0, 0, :T, :T].bool()
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask, dropout_p=0.0)
        
        # Reshape and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(resid_pdrop),
        ))
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