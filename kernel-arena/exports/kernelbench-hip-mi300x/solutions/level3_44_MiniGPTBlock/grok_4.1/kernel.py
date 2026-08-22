import os
os.environ["CXX"] = "hipcc"
from torch.utils.cpp_extension import load_inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NewGELU(nn.Module):
    def __init__(self):
        super(NewGELU, self).__init__()
    
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
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
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v 
        y = y.transpose(1, 2).contiguous().view(B, T, C) 

        y = self.resid_dropout(self.c_proj(y))
        return y
    
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

cpp_source = r"""
#include <hip/hip_runtime.h>
#include <cmath>

const float SQRT_2_OVER_PI = 0.7978845608028654f;

__global__ void layer_norm_kernel(
    const float *__restrict__ x,
    const float *__restrict__ gamma,
    const float *__restrict__ beta,
    float *__restrict__ y,
    int64_t B,
    int64_t T,
    int64_t C,
    float eps
) {
    int b_idx = blockIdx.x;
    int t_idx = blockIdx.y;
    if ((int64_t)b_idx >= B || (int64_t)t_idx >= T) return;

    int tid = threadIdx.x;
    int blk_sz = blockDim.x;
    extern __shared__ float shared_mem[];
    float* sum_x_shared = shared_mem;
    float* sum_xx_shared = shared_mem + blk_sz;

    float sum_x_local = 0.0f;
    float sum_xx_local = 0.0f;
    int64_t offset_base = (int64_t)b_idx * T * C + (int64_t)t_idx * C;
    for (int j = tid; j < (int)C; j += blk_sz) {
        float val = x[offset_base + j];
        sum_x_local += val;
        sum_xx_local += val * val;
    }

    sum_x_shared[tid] = sum_x_local;
    sum_xx_shared[tid] = sum_xx_local;
    __syncthreads();

    for (int s = blk_sz / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_x_shared[tid] += sum_x_shared[tid + s];
            sum_xx_shared[tid] += sum_xx_shared[tid + s];
        }
        __syncthreads();
    }

    float mean = sum_x_shared[0] / static_cast<float>(C);
    float xx_mean = sum_xx_shared[0] / static_cast<float>(C);
    float var = fmaxf(xx_mean - mean * mean, 0.0f);
    float scale = rsqrtf(var + eps);

    for (int j = tid; j < (int)C; j += blk_sz) {
        int64_t idx = offset_base + j;
        float val = x[idx];
        y[idx] = ((val - mean) * scale * gamma[j]) + beta[j];
    }
}

__global__ void new_gelu_kernel(const float *__restrict__ input, float *__restrict__ output, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];
        float x3 = x * x * x;
        float tanh_arg = SQRT_2_OVER_PI * (x + 0.044715f * x3);
        float tanh_out = tanhf(tanh_arg);
        output[idx] = 0.5f * x * (1.0f + tanh_out);
    }
}

torch::Tensor layer_norm_hip(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float eps = 1e-5f) {
    torch::Tensor out = torch::empty_like(x);
    int64_t B = x.size(0);
    int64_t T_len = x.size(1);
    int64_t C_len = x.size(2);
    const int threads = 256;
    dim3 grid(static_cast<int>(B), static_cast<int>(T_len));
    dim3 block(threads);
    size_t smem = 2 * threads * sizeof(float);
    layer_norm_kernel<<<grid, block, smem>>>(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        out.data_ptr<float>(),
        B, T_len, C_len, eps
    );
    return out;
}

torch::Tensor new_gelu_hip(torch::Tensor input) {
    torch::Tensor out = torch::empty_like(input);
    int64_t N = input.numel();
    const int block_size = 256;
    dim3 grid(static_cast<unsigned int>((N + block_size - 1LL) / block_size));
    dim3 threads(block_size);
    new_gelu_kernel<<<grid, threads>>>(
        input.data_ptr<float>(),
        out.data_ptr<float>(),
        N
    );
    return out;
}
"""

custom_ops = load_inline(
    name="gpt_block",
    cpp_sources=[cpp_source],
    functions=["layer_norm_hip", "new_gelu_hip"],
    verbose=True,
)

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
        self.custom_ops = custom_ops

    def mlpf(self, x):
        fc_out = self.mlp.c_fc(x)
        gelu_out = self.custom_ops.new_gelu_hip(fc_out)
        proj_out = self.mlp.c_proj(gelu_out)
        return self.mlp.dropout(proj_out)

    def forward(self, x):
        x1 = self.custom_ops.layer_norm_hip(x, self.ln_1.weight, self.ln_1.bias, 1e-5)
        x = x + self.attn(x1)
        x2 = self.custom_ops.layer_norm_hip(x, self.ln_2.weight, self.ln_2.bias, 1e-5)
        x = x + self.mlpf(x2)
        return x
