import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused transpose and reshape kernel: (B, nh, T, hs) -> (B, T, C)
fused_output_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Fused kernel: (B, nh, T, hs) -> (B, T, nh*hs) = (B, T, C)
__global__ void fused_transpose_and_reshape_kernel(
    const float* __restrict__ input,  // [B, nh, T, hs]
    float* __restrict__ output,       // [B, T, C]
    const int B,
    const int nh,
    const int T,
    const int hs,
    const int C
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * T * C;
    
    if (idx >= total) return;
    
    // Output layout: (B, T, C) where C = nh * hs
    const int i_c = idx % C;
    const int i_t = (idx / C) % T;
    const int i_b = idx / (C * T);
    
    // Map C to (nh, hs)
    const int i_hs = i_c % hs;
    const int i_nh = i_c / hs;
    
    // Input layout: (B, nh, T, hs)
    const int input_idx = i_b * (nh * T * hs) + i_nh * (T * hs) + i_t * hs + i_hs;
    
    output[idx] = input[input_idx];
}

torch::Tensor fused_transpose_reshape(torch::Tensor input) {
    // Input: (B, nh, T, hs)
    auto sizes = input.sizes();
    int B = sizes[0];
    int nh = sizes[1];
    int T = sizes[2];
    int hs = sizes[3];
    int C = nh * hs;
    
    auto output = torch::empty({B, T, C}, input.options());
    
    int total = B * T * C;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    
    fused_transpose_and_reshape_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, nh, T, hs, C
    );
    
    return output;
}
"""

fused_output_cpp = """
torch::Tensor fused_transpose_reshape(torch::Tensor input);
"""

fused_output_module = load_inline(
    name="fused_output_v7",
    cpp_sources=fused_output_cpp,
    cuda_sources=fused_output_source,
    functions=["fused_transpose_reshape"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    """
    Optimized multi-head masked self-attention using PyTorch's efficient SDPA.
    """

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
        self.attn_pdrop = attn_pdrop
        self.fused_output = fused_output_module

    def forward(self, x):
        B, T, C = x.size()

        # QKV projection
        qkv = self.c_attn(x)  # (B, T, 3*C)
        
        # Efficient reshape for multi-head attention
        qkv = qkv.view(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Use PyTorch's optimized scaled_dot_product_attention
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None,
            dropout_p=self.attn_pdrop if self.training else 0.0,
            is_causal=True
        )
        
        # Fused transpose and reshape
        y = self.fused_output.fused_transpose_reshape(y.contiguous())

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


def custom_kernel(inputs):
    """Entry point for wafer evaluation"""
    n_embd = 768
    n_head = 8
    attn_pdrop = 0.0
    resid_pdrop = 0.0
    max_seqlen = 1024
    
    model = ModelNew(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen).cuda()
    model.eval()
    
    x = inputs[0]
    with torch.no_grad():
        return model(x)
