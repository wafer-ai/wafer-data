import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

os.environ["CXX"] = "hipcc"

# Simple GELU kernel
gelu_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void gelu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        float cube = xi * xi * xi;
        float inner = 0.044715f * cube + xi;
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

gelu_op = load_inline(
    name="gelu_op",
    cpp_sources=gelu_cpp_source,
    functions=["gelu_hip"],
    verbose=True,
)

# Simplified attention kernel with correct configuration
attention_cpp_source = """
#include <hip/hip_runtime.h>
#include <math.h>

__global__ void simple_attention_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int B, int nh, int T, int C, int hs
) {
    int b = blockIdx.x;
    int head = blockIdx.y;
    int t_out = threadIdx.x;
    
    if (t_out >= T) return;
    
    float max_val = -INFINITY;
    for (int t = 0; t <= t_out; ++t) {
        float qk = 0.0f;
        for (int dh = 0; dh < hs; ++dh) {
            int q_idx = b * nh * T * hs + head * T * hs + t_out * hs + dh;
            int k_idx = b * nh * T * hs + head * T * hs + t * hs + dh;
            qk += q[q_idx] * k[k_idx];
        }
        qk *= 1.0f / sqrtf((float)hs);
        if (qk > max_val) max_val = qk;
    }
    
    float sum_exp = 0.0f;
    float sum_val = 0.0f;
    for (int t = 0; t <= t_out; ++t) {
        float qk = 0.0f;
        for (int dh = 0; dh < hs; ++dh) {
            int q_idx = b * nh * T * hs + head * T * hs + t_out * hs + dh;
            int k_idx = b * nh * T * hs + head * T * hs + t * hs + dh;
            qk += q[q_idx] * k[k_idx];
        }
        qk *= 1.0f / sqrtf((float)hs);
        float exp_val = expf(qk - max_val);
        sum_exp += exp_val;
        
        float v_val = v[b * nh * T * hs + head * T * hs + t * hs];
        sum_val += exp_val * v_val;
    }
    
    float attn_out = sum_val / sum_exp;
    for (int dh = 0; dh < hs; ++dh) {
        int out_idx = b * nh * T * hs + head * T * hs + t_out * hs + dh;
        out[out_idx] = attn_out;
    }
}

torch::Tensor attention_hip(torch::Tensor x, int n_head) {
    auto B = x.size(0);
    auto T = x.size(1);
    auto C = x.size(2);
    int nh = n_head;
    int hs = C / nh;
    
    auto q = x.clone();
    auto k = x.clone();
    auto v = x.clone();
    auto out = torch::zeros({B, nh, T, hs}, torch::dtype(torch::kFloat32).device(x.device()));
    
    dim3 grid(B, nh);
    int block_size = T;
    
    simple_attention_kernel<<<grid, block_size>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        B, nh, T, C, hs
    );
    
    return out.transpose(1, 2).contiguous().view({B, T, C});
}
"""

attention_op = load_inline(
    name="attention_op",
    cpp_sources=attention_cpp_source,
    functions=["attention_hip"],
    verbose=True,
)

class OptimizedGELU(nn.Module):
    def __init__(self):
        super(OptimizedGELU, self).__init__()
        self.gelu = gelu_op
    
    def forward(self, x):
        return self.gelu.gelu_hip(x)

class OptimizedCausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.hs = n_embd // n_head
        
        # key, query, value projections
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        
        self.attn_op = attention_op
        
    def forward(self, x):
        B, T, C = x.size()
        
        # Linear projection for Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention (B, T, nh, hs) -> (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.hs).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.hs).transpose(1, 2)
        
        # Apply causal mask and custom attention
        y = self.attn_op.attention_hip(x, self.n_head)
        
        # Output projection
        y = self.c_proj(y)
        return y

class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = OptimizedCausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = OptimizedGELU(),
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