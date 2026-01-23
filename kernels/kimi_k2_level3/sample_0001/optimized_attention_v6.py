import os
import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Set compiler to hipcc for AMD GPUs
os.environ["CXX"] = "hipcc"

# Simple but optimized kernel to fuse masking with softmax computation
attention_optimized_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void elementwise_causal_mask_kernel(
    float* attn_scores,
    const float* bias_mask,
    int B, int nh, int T, int seq_len
) {
    int total_elements = B * nh * T * T;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;
    
    // Calculate 4D indices from flat index
    int b = idx / (nh * T * T);
    int rem = idx % (nh * T * T);
    int h = rem / (T * T);
    int rem2 = rem % (T * T);
    int row = rem2 / T;
    int col = rem2 % T;
    
    if (col > row || row >= seq_len) {
        attn_scores[idx] = -INFINITY;
    }
}

torch::Tensor optimized_attention_forward(
    torch::Tensor q, 
    torch::Tensor k, 
    torch::Tensor v,
    torch::Tensor bias_mask,
    float scale
) {
    auto B = q.size(0);
    auto nh = q.size(1);
    auto T = q.size(2);
    auto hs = q.size(3);
    
    // Reshape tensors for batch matrix multiplication
    auto q_flat = q.reshape({B * nh, T, hs});
    auto k_flat = k.reshape({B * nh, T, hs});
    auto v_flat = v.reshape({B * nh, T, hs});
    
    // Step 1: Q @ K^T using optimized BLAS
    auto attn_scores = torch::bmm(q_flat, k_flat.transpose(-2, -1));
    attn_scores = attn_scores * scale;
    attn_scores = attn_scores.reshape({B, nh, T, T});
    
    // Step 2: Apply causal mask using efficient elementwise kernel
    int total_elements = B * nh * T * T;
    const int threads_per_block = 256;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    elementwise_causal_mask_kernel<<<num_blocks, threads_per_block>>>(
        attn_scores.data_ptr<float>(),
        bias_mask.data_ptr<float>(),
        B, nh, T, T
    );
    
    // Step 3: Softmax using PyTorch's highly optimized implementation
    auto attn_weights = attn_scores.softmax(-1);
    
    // Step 4: attn_weights @ V using optimized BLAS
    auto attn_weights_flat = attn_weights.reshape({B * nh, T, T});
    auto out = torch::bmm(attn_weights_flat, v_flat);
    out = out.reshape({B, nh, T, hs});
    
    return out;
}
"""

# Compile the custom kernel
attention_optimized = load_inline(
    name="attention_optimized",
    cpp_sources=attention_optimized_cpp_source,
    functions=["optimized_attention_forward"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.register_buffer("bias_mask", torch.triu(torch.full((max_seqlen, max_seqlen), float('-inf')), diagonal=1).view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd
        self.attn_pdrop = attn_pdrop
        self.optimized_attention = attention_optimized

    def forward(self, x):
        B, T, C = x.size()
        
        # Calculate query, key, values
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention - use permute instead of view+transpose
        k = k.view(B, T, self.n_head, C // self.n_head).permute(0, 2, 1, 3).contiguous()
        q = q.view(B, T, self.n_head, C // self.n_head).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, T, self.n_head, C // self.n_head).permute(0, 2, 1, 3).contiguous()
        
        # Use optimized attention kernel with fused masking
        scale = 1.0 / math.sqrt(k.size(-1))
        y = self.optimized_attention.optimized_attention_forward(
            q, k, v, self.bias_mask[:, :, :T, :T], scale
        )
        
        # Reshape output
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        
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
    return [torch.randn(batch_size, seq_len, n_embd, device='cuda')]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]