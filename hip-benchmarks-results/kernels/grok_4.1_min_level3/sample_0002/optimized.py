import os
os.environ["CXX"] = "hipcc"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

new_gelu_cpp = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__global__ void new_gelu_kernel(const float *input, float *output, int64_t size) {
  int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < size) {
    float x = input[idx];
    float x3 = x * x * x;
    const float PI = 3.141592653589793f;
    const float coeff = sqrtf(2.0f / PI);
    const float cd = 0.044715f;
    float inner = x + cd * x3;
    float tanh_out = tanhf(coeff * inner);
    output[idx] = 0.5f * x * (1.0f + tanh_out);
  }
}

torch::Tensor new_gelu_hip(torch::Tensor input) {
  torch::Tensor output = torch::zeros_like(input);
  int64_t size = input.numel();
  const int threads = 256;
  dim3 block(threads);
  dim3 grid((size + threads - 1) / threads);
  new_gelu_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
  return output;
}
"""

new_gelu = load_inline(
    name="new_gelu",
    cpp_sources=new_gelu_cpp,
    functions=["new_gelu_hip"],
    verbose=True,
)

class NewGELU(nn.Module):
    def __init__(self):
        super(NewGELU, self).__init__()

    def forward(self, x):
        return new_gelu.new_gelu_hip(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd
        self.scale = 1.0 / math.sqrt(n_embd / self.n_head)

    def forward(self, x):
        B, T, C = x.size() 
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 

        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v 
        y = y.transpose(1, 2).contiguous().view(B, T, C) 

        y = self.c_proj(y)
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
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x)))

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
    return [torch.rand(batch_size, seq_len, n_embd)]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]
