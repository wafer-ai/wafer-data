import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused QKV projection + split + reshape kernel for better memory locality
fused_qkv_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Fused transpose reshape kernel: (B, T, nh, hs) -> (B, nh, T, hs)
__global__ void fused_transpose_reshape_kernel(
    const float* __restrict__ input,  // [B, T, nh, hs]
    float* __restrict__ output,       // [B, nh, T, hs]
    const int B,
    const int T,
    const int nh,
    const int hs
) {
    const int total_elems = B * nh * T * hs;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elems) return;
    
    // Decode output index
    const int i_hs = idx % hs;
    const int i_T = (idx / hs) % T;
    const int i_nh = (idx / (hs * T)) % nh;
    const int i_B = idx / (hs * T * nh);
    
    // Calculate input index for (B, T, nh, hs) layout
    const int input_idx = i_B * (T * nh * hs) + i_T * (nh * hs) + i_nh * hs + i_hs;
    
    output[idx] = input[input_idx];
}

std::vector<torch::Tensor> fused_qkv_transpose(
    torch::Tensor qkv,  // [B, T, 3*n_embd]
    int n_embd,
    int n_head
) {
    auto sizes = qkv.sizes();
    int B = sizes[0];
    int T = sizes[1];
    int hs = n_embd / n_head;
    
    // Split into q, k, v
    auto chunks = qkv.split(n_embd, /*dim=*/2);
    auto q_flat = chunks[0].view({B, T, n_head, hs}).contiguous();
    auto k_flat = chunks[1].view({B, T, n_head, hs}).contiguous();
    auto v_flat = chunks[2].view({B, T, n_head, hs}).contiguous();
    
    // Output tensors [B, nh, T, hs]
    auto q = torch::empty({B, n_head, T, hs}, qkv.options());
    auto k = torch::empty({B, n_head, T, hs}, qkv.options());
    auto v = torch::empty({B, n_head, T, hs}, qkv.options());
    
    int total_elems = B * n_head * T * hs;
    int block_size = 256;
    int num_blocks = (total_elems + block_size - 1) / block_size;
    
    fused_transpose_reshape_kernel<<<num_blocks, block_size>>>(
        q_flat.data_ptr<float>(), q.data_ptr<float>(), B, T, n_head, hs);
    fused_transpose_reshape_kernel<<<num_blocks, block_size>>>(
        k_flat.data_ptr<float>(), k.data_ptr<float>(), B, T, n_head, hs);
    fused_transpose_reshape_kernel<<<num_blocks, block_size>>>(
        v_flat.data_ptr<float>(), v.data_ptr<float>(), B, T, n_head, hs);
    
    return {q, k, v};
}
"""

fused_qkv_cpp = """
std::vector<torch::Tensor> fused_qkv_transpose(torch::Tensor qkv, int n_embd, int n_head);
"""

fused_qkv_module = load_inline(
    name="fused_qkv_v5",
    cpp_sources=fused_qkv_cpp,
    cuda_sources=fused_qkv_source,
    functions=["fused_qkv_transpose"],
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
        self.attn_pdrop = attn_pdrop
        self.fused_qkv = fused_qkv_module

    def forward(self, x):
        B, T, C = x.size()
        hs = C // self.n_head

        # QKV projection
        qkv = self.c_attn(x)
        
        # Fused split and transpose
        q, k, v = self.fused_qkv.fused_qkv_transpose(qkv, self.n_embd, self.n_head)

        # Use PyTorch's optimized scaled_dot_product_attention with causal mask
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None,
            dropout_p=self.attn_pdrop if self.training else 0.0,
            is_causal=True
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
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
