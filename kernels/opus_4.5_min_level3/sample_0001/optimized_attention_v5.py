import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused QKV projection + reshape kernel
fused_qkv_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Fused kernel for reshaping Q, K, V from [B, T, 3*C] to separate [B, nh, T, hs] tensors
// This avoids multiple memory traversals
__global__ void fused_qkv_reshape_kernel(
    const float* __restrict__ qkv,      // [B, T, 3*C]
    float* __restrict__ q_out,           // [B, nh, T, hs]
    float* __restrict__ k_out,           // [B, nh, T, hs]
    float* __restrict__ v_out,           // [B, nh, T, hs]
    int B, int T, int C, int nh, int hs
) {
    // Each thread handles one element
    int total = B * nh * T * hs;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total) return;
    
    // Decode output index: [b, h, t, s]
    int s = idx % hs;
    int t = (idx / hs) % T;
    int h = (idx / (hs * T)) % nh;
    int b = idx / (hs * T * nh);
    
    // Input index in [B, T, 3*C] format
    // Q is at offset 0, K at offset C, V at offset 2*C
    int qkv_idx_base = b * T * 3 * C + t * 3 * C + h * hs + s;
    
    q_out[idx] = qkv[qkv_idx_base];
    k_out[idx] = qkv[qkv_idx_base + C];
    v_out[idx] = qkv[qkv_idx_base + 2 * C];
}

std::vector<torch::Tensor> fused_qkv_reshape_hip(torch::Tensor qkv, int nh) {
    auto B = qkv.size(0);
    auto T = qkv.size(1);
    auto C3 = qkv.size(2);
    auto C = C3 / 3;
    auto hs = C / nh;
    
    auto q = torch::empty({B, nh, T, hs}, qkv.options());
    auto k = torch::empty({B, nh, T, hs}, qkv.options());
    auto v = torch::empty({B, nh, T, hs}, qkv.options());
    
    int total = B * nh * T * hs;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    
    fused_qkv_reshape_kernel<<<num_blocks, block_size>>>(
        qkv.data_ptr<float>(),
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        B, T, C, nh, hs
    );
    
    return {q, k, v};
}
"""

cpp_source = """
std::vector<torch::Tensor> fused_qkv_reshape_hip(torch::Tensor qkv, int nh);
"""

fused_qkv = load_inline(
    name="fused_qkv_reshape",
    cpp_sources=cpp_source,
    cuda_sources=fused_qkv_source,
    functions=["fused_qkv_reshape_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
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
        self.attn_pdrop = attn_pdrop
        self.fused_qkv = fused_qkv

    def forward(self, x):
        B, T, C = x.size()

        # QKV projection
        qkv = self.c_attn(x)
        
        # Fused reshape and split
        q, k, v = self.fused_qkv.fused_qkv_reshape_hip(qkv, self.n_head)

        # Use scaled_dot_product_attention with causal mask - leverages Flash Attention
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_pdrop if self.training else 0.0,
            is_causal=True
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


def get_inputs():
    return [torch.rand(128, 512, 768).cuda()]


def get_init_inputs():
    return [768, 8, 0.0, 0.0, 1024]
