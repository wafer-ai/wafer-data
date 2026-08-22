import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized softmax kernel with causal masking built-in
masked_softmax_source = """
#include <hip/hip_runtime.h>

#define BLOCK_SIZE 128

__global__ void masked_softmax_kernel(
    float* attn,
    int batch_size, int num_heads, int seq_len) {
    
    int b = blockIdx.z;  // Batch
    int h = blockIdx.y;  // Head
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Row
    
    if (i >= seq_len) return;
    
    // Base for this attention row
    int row_base = (b * num_heads * seq_len + h * seq_len + i) * seq_len;
    
    // Find max
    float max_val = -1e20f;
    for (int j = 0; j < seq_len; j++) {
        float val = attn[row_base + j];
        max_val = fmaxf(max_val, val);
    }
    
    // Sum exp
    float sum_exp = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        float val = attn[row_base + j];
        sum_exp += expf(val - max_val);
    }
    
    // Apply softmax
    float inv_sum = 1.0f / (sum_exp + 1e-6f);
    for (int j = 0; j < seq_len; j++) {
        float val = attn[row_base + j];
        attn[row_base + j] = expf(val - max_val) * inv_sum;
    }
}

torch::Tensor masked_softmax_hip(torch::Tensor attn) {
    auto batch_size = attn.size(0);
    auto num_heads = attn.size(1);
    auto seq_len = attn.size(2);
    
    int block_dim = BLOCK_SIZE;
    dim3 grid_dim((seq_len + block_dim - 1) / block_dim, num_heads, batch_size);
    
    masked_softmax_kernel<<<grid_dim, block_dim>>>(
        attn.data_ptr<float>(),
        batch_size, num_heads, seq_len
    );
    
    return attn;
}
"""

masked_softmax = load_inline(
    name="masked_softmax",
    cpp_sources=masked_softmax_source,
    functions=["masked_softmax_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    An optimized multi-head masked self-attention layer using custom HIP kernels.
    Optimizes the softmax operation with integrated causal masking.
    """
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd
        
        # Custom masked softmax kernel
        self.masked_softmax = masked_softmax

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        
        # Use custom masked softmax kernel
        att = self.masked_softmax.masked_softmax_hip(att)
        
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y