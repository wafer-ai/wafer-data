import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

os.environ["CXX"] = "hipcc"


class ModelNew(nn.Module):
    """
    Optimized multi-head masked self-attention using PyTorch's efficient SDPA.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # Fused QKV projection
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.attn_pdrop = attn_pdrop

    def forward(self, x):
        B, T, C = x.size()

        # Single fused QKV projection
        qkv = self.c_attn(x)  # (B, T, 3*C)
        
        # Efficient reshape and permute using a single view+permute chain
        qkv = qkv.view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Use direct indexing instead of unbind

        # Use PyTorch's optimized scaled_dot_product_attention with causal mask
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None,
            dropout_p=self.attn_pdrop if self.training else 0.0,
            is_causal=True
        )
        
        # Efficient reshape back using contiguous view
        y = y.permute(0, 2, 1, 3).reshape(B, T, C)

        # Output projection with residual dropout
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
