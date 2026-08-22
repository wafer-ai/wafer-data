import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

os.environ["CXX"] = "hipcc"


class ModelNew(nn.Module):
    """
    Optimized multi-head masked self-attention using PyTorch's efficient SDPA.
    Uses contiguous memory layout and efficient view operations.
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

        # Single fused QKV projection - most efficient memory access pattern
        qkv = self.c_attn(x)  # (B, T, 3*C)
        
        # Reshape and split for multi-head attention
        # Shape: (B, T, 3, nh, hs)
        qkv = qkv.view(B, T, 3, self.n_head, self.head_dim)
        
        # Permute to (3, B, nh, T, hs) and unbind
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each is (B, nh, T, hs)

        # Use PyTorch's optimized scaled_dot_product_attention with causal mask
        # This uses flash attention or memory-efficient attention on supported hardware
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None,
            dropout_p=self.attn_pdrop if self.training else 0.0,
            is_causal=True
        )
        
        # Reshape back: (B, nh, T, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

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
