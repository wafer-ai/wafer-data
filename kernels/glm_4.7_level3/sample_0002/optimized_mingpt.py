import os
import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# GELU activation kernel
gelu_source = """
#include <hip/hip_runtime.h>

__global__ void gelu_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float x3 = val * val * val;
        float tanh_arg = 0.7978845608028654f * (val + 0.044715f * x3);
        x[idx] = 0.5f * val * (1.0f + tanhf(tanh_arg));
    }
}

void gelu_hip(torch::Tensor x) {
    auto size = x.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), size);
}
"""

gelu_module = load_inline(
    name="gelu_activation",
    cpp_sources=gelu_source,
    functions=["gelu_hip"],
    verbose=True,
)


class NewGELUOptimized(nn.Module):
    """
    Optimized GELU activation with HIP kernel
    """
    def __init__(self, gelu_kernel_module):
        super().__init__()
        self.gelu_kernel = gelu_kernel_module
    
    def forward(self, x):
        y = x.clone()  # Don't modify in-place
        self.gelu_kernel.gelu_hip(y)
        return y


class CausalSelfAttentionNew(nn.Module):
    """
    Keep standard PyTorch attention for correctness
    """
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()
        
        # Standard PyTorch attention
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = torch.nn.functional.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        
        return y


class ModelNew(nn.Module):
    """Optimized MiniGPT block with optimized GELU activation."""
    
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.gelu_module = gelu_module
        
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttentionNew(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = NewGELUOptimized(self.gelu_module),
            dropout = nn.Dropout(resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x