import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused output reshape kernel - from [B, nh, T, hs] to [B, T, C]
fused_output_reshape_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Fused kernel for reshaping attention output from [B, nh, T, hs] to [B, T, C]
// Input is contiguous in memory as [B, nh, T, hs]
__global__ void fused_output_reshape_kernel(
    const float* __restrict__ input,    // [B, nh, T, hs] contiguous
    float* __restrict__ output,          // [B, T, C] contiguous
    int B, int T, int nh, int hs
) {
    int C = nh * hs;
    int total = B * T * C;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total) return;
    
    // Decode output index: [b, t, c] where c = h * hs + s
    int c = idx % C;
    int t = (idx / C) % T;
    int b = idx / (C * T);
    
    int h = c / hs;
    int s = c % hs;
    
    // Input index in [B, nh, T, hs] format (contiguous layout)
    int in_idx = b * (nh * T * hs) + h * (T * hs) + t * hs + s;
    
    output[idx] = input[in_idx];
}

torch::Tensor fused_output_reshape_hip(torch::Tensor input, int B, int T, int C) {
    // Ensure input is contiguous
    auto input_contig = input.contiguous();
    auto output = torch::empty({B, T, C}, input_contig.options());
    
    int total = B * T * C;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    
    int nh = input_contig.size(1);
    int hs = input_contig.size(3);
    
    fused_output_reshape_kernel<<<num_blocks, block_size>>>(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        B, T, nh, hs
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_output_reshape_hip(torch::Tensor input, int B, int T, int C);
"""

fused_output = load_inline(
    name="fused_output_reshape_v2",
    cpp_sources=cpp_source,
    cuda_sources=fused_output_reshape_source,
    functions=["fused_output_reshape_hip"],
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
        self.fused_output = fused_output

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Use scaled_dot_product_attention with causal mask - leverages Flash Attention
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_pdrop if self.training else 0.0,
            is_causal=True
        )

        # Fused transpose and reshape (input is [B, nh, T, hs], output is [B, T, C])
        y = self.fused_output.fused_output_reshape_hip(y, B, T, C)
        
        y = self.resid_dropout(self.c_proj(y))
        return y


def get_inputs():
    return [torch.rand(128, 512, 768).cuda()]


def get_init_inputs():
    return [768, 8, 0.0, 0.0, 1024]
