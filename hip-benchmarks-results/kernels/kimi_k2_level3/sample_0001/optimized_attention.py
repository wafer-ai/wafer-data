import os
import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Set compiler to hipcc for AMD GPUs
os.environ["CXX"] = "hipcc"

# Custom HIP kernel that fuses masking, softmax, and dropout
attention_fused_cpp_source = """
#include <hip/hip_runtime.h>
#include <math.h>

// Fused kernel: masked softmax with dropout
// This reduces memory traffic by combining three operations into one kernel
__global__ void masked_softmax_dropout_kernel(
    float* attn_scores, float* attn_weights,
    const float* bias, float dropout_prob,
    int B, int nh, int T, unsigned long long seed
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int row = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (row >= T) return;
    
    float* row_scores = &attn_scores[batch_idx * nh * T * T + head_idx * T * T + row * T];
    float* row_weights = &attn_weights[batch_idx * nh * T * T + head_idx * T * T + row * T];
    const float* row_bias = &bias[row * T];
    
    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int col = 0; col <= row; col++) {
        float val = row_scores[col];
        max_val = fmaxf(max_val, val);
    }
    
    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int col = 0; col <= row; col++) {
        float exp_val = expf(row_scores[col] - max_val);
        sum_exp += exp_val;
        row_weights[col] = exp_val;
    }
    
    // Normalize and apply dropout and causal mask
    for (int col = 0; col < T; col++) {
        float prob;
        if (col <= row) {
            prob = row_weights[col] / sum_exp;
            
            // Simple deterministic dropout based on position
            unsigned long long seed_val = seed + batch_idx * nh * T * T + head_idx * T * T + row * T + col;
            float random_val = (float)((seed_val * 0x5DEECE66DLL + 0xB) & 0xFFFFFFFFFFFFF) / (float)0xFFFFFFFFFFFFF;
            
            if (random_val < dropout_prob) {
                prob = 0.0f;
            }
        } else {
            // Causal mask
            prob = 0.0f;
        }
        row_weights[col] = prob;
    }
}

torch::Tensor fused_attention_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor bias, float dropout_prob, float scale
) {
    auto B = q.size(0);
    auto nh = q.size(1);
    auto T = q.size(2);
    auto hs = q.size(3);
    
    // Compute Q @ K^T (uses highly optimized BLAS)
    auto attn_scores = torch::bmm(
        q.view({B * nh, T, hs}),
        k.view({B * nh, T, hs}).transpose(-2, -1)
    ).view({B, nh, T, T}) * scale;
    
    // Allocate attention weights
    auto attn_weights = torch::empty_like(attn_scores);
    
    // Launch fused kernel
    dim3 block(32); // 32 threads per row for efficient execution
    dim3 grid(B, nh, (T + block.x - 1) / block.x);
    
    // Fixed seed for reproducibility
    unsigned long long seed = 42;
    
    masked_softmax_dropout_kernel<<<grid, block>>>(
        attn_scores.data_ptr<float>(),
        attn_weights.data_ptr<float>(),
        bias.data_ptr<float>(),
        dropout_prob,
        B, nh, T, seed
    );
    
    // Compute attn_weights @ v (uses highly optimized BLAS)
    auto out = torch::bmm(
        attn_weights.view({B * nh, T, T}),
        v.view({B * nh, T, hs})
    ).view({B, nh, T, hs});
    
    return out;
}
"""

# Compile the custom kernel
attention_fused = load_inline(
    name="attention_fused",
    cpp_sources=attention_fused_cpp_source,
    functions=["fused_attention_forward"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd
        self.attn_pdrop = attn_pdrop
        self.fused_attention = attention_fused

    def forward(self, x):
        B, T, C = x.size()
        
        # Calculate query, key, values
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Use fused attention kernel
        scale = 1.0 / math.sqrt(k.size(-1))
        y = self.fused_attention.fused_attention_forward(
            q, k, v, self.bias[:, :, :T, :T], self.attn_pdrop, scale
        )
        
        # Reshape output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        
        return y

# Model configuration
batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0

def get_inputs():
    return [torch.rand(batch_size, seq_len, n_embd, device='cuda')]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]