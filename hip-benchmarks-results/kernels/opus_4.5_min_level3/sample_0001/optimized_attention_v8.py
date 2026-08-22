import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

os.environ["CXX"] = "hipcc"


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

    def forward(self, x):
        B, T, C = x.size()

        # Use chunk instead of split for potentially better performance
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=2)
        
        # Reshape and transpose in a single operation
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2).contiguous()
        k = k.view(B, T, self.n_head, hs).transpose(1, 2).contiguous()
        v = v.view(B, T, self.n_head, hs).transpose(1, 2).contiguous()

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
