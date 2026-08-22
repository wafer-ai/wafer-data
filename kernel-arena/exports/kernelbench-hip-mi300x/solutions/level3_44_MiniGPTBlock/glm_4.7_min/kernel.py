import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NewGELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5*x*(1.0+torch.tanh(math.sqrt(2.0/math.pi)*(x+0.044715*x*x*x)))

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, maxseqlen):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3*n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.register_buffer("bias", torch.tril(torch.ones(maxseqlen,maxseqlen)).view(1,1,maxseqlen,maxseqlen))
        self.n_head = n_head
        self.n_embd = n_embd
    def forward(self, x):
        B,T,C = x.size()
        q,k,v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        att = (q@k.transpose(-2,-1))/math.sqrt(C//self.n_head)
        att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        att = F.softmax(att,-1)
        att = self.attn_drop(att)
        y = (att@v).transpose(1,2).contiguous().view(B,T,C)
        return self.resid_drop(self.c_proj(y))

class Model(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, maxseqlen):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd,n_head,attn_pdrop,resid_pdrop,maxseqlen)
        self.ln2 = nn.LayerNorm(n_embd)
        mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(n_embd,4*n_embd),
            c_proj=nn.Linear(4*n_embd,n_embd),
            act=NewGELU(),
            drop=nn.Dropout(resid_pdrop)
        ))
        self.mlp = lambda x: mlp.drop(mlp.c_proj(mlp.act(mlp.c_fc(x))))
    def forward(self, x):
        return x + self.mlp(self.ln2(x + self.attn(self.ln1(x))))

ModelNew = Model