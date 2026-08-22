import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

os.environ["CXX"] = "hipcc"

# Highly optimized GELU kernel
gelu_cpp_source = """
#include <hip/hip_runtime.h>
#define GELU_SCALING 0.044715f
#define SQRT_2_OVER_PI 0.7978845608028654f  // sqrt(2.0 / M_PI)

__global__ void gelu_fast_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float x3 = x * x * x;
        float inner = GELU_SCALING * x3 + x;
        float tanh_val = tanhf(SQRT_2_OVER_PI * inner);
        output[idx] = 0.5f * x * (1.0f + tanh_val);
    }
}

torch::Tensor gelu_hip(torch::Tensor x) {
    int n = x.numel();
    auto out = torch::zeros_like(x);
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;
    gelu_fast_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""

gelu_op = load_inline(
    name="gelu_op",
    cpp_sources=gelu_cpp_source,
    functions=["gelu_hip"],
    verbose=True,
)

# Fast MLP with fused operations
class OptimizedMLP(nn.Module):
    def __init__(self, n_embd, resid_pdrop):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.act = gelu_op
        self.dropout = nn.Dropout(resid_pdrop)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.act.gelu_hip(x)
        x = self.c_proj(x)
        return self.dropout(x)

# Use PyTorch's optimized MultiheadAttention
class OptimizedCausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        
        # Use PyTorch's built-in MHA (which uses optimized kernels)
        self.mha = nn.MultiheadAttention(n_embd, n_head, dropout=attn_pdrop, batch_first=True)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        
    def forward(self, x):
        B, T, C = x.size()
        
        # PyTorch's optimized multi-head attention
        attn_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        y, _ = self.mha(x, x, x, attn_mask=attn_mask, need_weights=False)
        
        return self.resid_dropout(self.c_proj(y))

class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = OptimizedCausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = OptimizedMLP(n_embd, resid_pdrop)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
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