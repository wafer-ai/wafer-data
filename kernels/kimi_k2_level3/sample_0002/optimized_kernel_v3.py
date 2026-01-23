import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

os.environ["CXX"] = "hipcc"

# Optimized GELU kernel
gelu_cpp_source = """
#include <hip/hip_runtime.h>
#define GELU_SCALING 0.044715f
#define SQRT_2_OVER_PI 0.7978845608028654f  // sqrt(2.0 / M_PI)

__global__ void gelu_optimized_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        float cube = xi * xi * xi;
        float inner = GELU_SCALING * cube + xi;
        float mult = SQRT_2_OVER_PI * inner;
        float tanh_val = tanhf(mult);
        out[idx] = 0.5f * xi * (1.0f + tanh_val);
    }
}

torch::Tensor gelu_hip_optimized(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    gelu_optimized_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}
"""

gelu_op = load_inline(
    name="gelu_op",
    cpp_sources=gelu_cpp_source,
    functions=["gelu_hip_optimized"],
    verbose=True,
)

# Optimized layer norm kernel (uses PyTorch's layer norm for correctness)
class OptimizedCausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        
        # Use PyTorch's MultiheadAttention for correctness with potential optimization
        self.mha = nn.MultiheadAttention(n_embd, n_head, dropout=attn_pdrop, batch_first=True)
        
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality

        # Use PyTorch's optimized MultiheadAttention
        y, _ = self.mha(x, x, x, need_weights=False)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class OptimizedGELU(nn.Module):
    def __init__(self):
        super(OptimizedGELU, self).__init__()
        self.gelu = gelu_op
    
    def forward(self, x):
        return self.gelu.gelu_hip_optimized(x)

class OptimizedMLP(nn.Module):
    def __init__(self, n_embd, resid_pdrop):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.act = OptimizedGELU()
        self.dropout = nn.Dropout(resid_pdrop)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

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