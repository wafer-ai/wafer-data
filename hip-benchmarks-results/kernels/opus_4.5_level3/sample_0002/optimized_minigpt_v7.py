import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Only GELU kernel - LayerNorm is already optimized in PyTorch
hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// GELU with float4 vectorization
__global__ void gelu_vec4_kernel(const float4* __restrict__ input, 
                                  float4* __restrict__ output, 
                                  int n) {
    const float c1 = 0.7978845608028654f;  // sqrt(2/pi)
    const float c2 = 0.044715f;
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float4 v = input[i];
        
        float x, y, z, w;
        x = v.x; v.x = 0.5f * x * (1.0f + tanhf(c1 * (x + c2 * x * x * x)));
        y = v.y; v.y = 0.5f * y * (1.0f + tanhf(c1 * (y + c2 * y * y * y)));
        z = v.z; v.z = 0.5f * z * (1.0f + tanhf(c1 * (z + c2 * z * z * z)));
        w = v.w; v.w = 0.5f * w * (1.0f + tanhf(c1 * (w + c2 * w * w * w)));
        
        output[i] = v;
    }
}

// Scalar GELU for remainder
__global__ void gelu_scalar_kernel(const float* __restrict__ input, 
                                    float* __restrict__ output,
                                    int start, int n) {
    const float c1 = 0.7978845608028654f;
    const float c2 = 0.044715f;
    
    int i = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (i < n) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + tanhf(c1 * (x + c2 * x * x * x)));
    }
}

torch::Tensor fused_gelu(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    int size4 = size / 4;
    int remainder = size % 4;
    
    if (size4 > 0) {
        const int block = 256;
        const int grid = (size4 + block - 1) / block;
        
        gelu_vec4_kernel<<<grid, block>>>(
            reinterpret_cast<const float4*>(input.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            size4
        );
    }
    
    if (remainder > 0) {
        const int block = 256;
        const int grid = (remainder + block - 1) / block;
        int start = size4 * 4;
        
        gelu_scalar_kernel<<<grid, block>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            start, size
        );
    }
    
    return output;
}
"""

custom_ops = load_inline(
    name="custom_ops_v7",
    cpp_sources=hip_source,
    functions=["fused_gelu"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

    def forward(self, x):
        B, T, C = x.size()
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Flash Attention
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        h = self.ln_2(x)
        h = self.c_fc(h)
        h = custom_ops.fused_gelu(h)
        h = self.c_proj(h)
        h = self.dropout(h)
        x = x + h
        return x


def custom_kernel(inputs):
    x = inputs[0]
    n_embd = 768
    n_head = 8
    attn_pdrop = 0.0
    resid_pdrop = 0.0
    max_seqlen = 1024
    
    model = ModelNew(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen).cuda()
    model.eval()
    
    with torch.no_grad():
        return model(x)
