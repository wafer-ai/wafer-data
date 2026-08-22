import torch
import torch.nn as nn
import torch.nn.functional as F
import math

batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0

class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.n_embd = n_embd
        self.n_head_size = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_head_size)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x).split(self.n_embd, dim=2)
        q, k, v = [yy.view(B, T, self.n_head, self.n_head_size).transpose(1, 2) for yy in qkv]
        y_heads = F.scaled_dot_product_attention(q, k, v, scale=self.scale, is_causal=True)
        y = y_heads.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

def get_inputs():
    return [torch.rand(batch_size, seq_len, n_embd)]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]