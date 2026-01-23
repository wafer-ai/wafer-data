import os
import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Set compiler to hipcc for AMD GPUs
os.environ["CXX"] = "hipcc"

# Custom HIP kernel that fuses masking and softmax operations
attention_fused_cpp_source = """
#include <hip/hip_runtime.h>
#include <math.h>

__global__ void masked_softmax_kernel(
    float* attn_scores, // Modified in-place
    const float* bias,
    int B, int nh, int T
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    if (row >= T || col >= T) return;
    
    // Calculate indices
    int idx = batch_idx * nh * T * T + head_idx * T * T + row * T + col;
    
    float score = attn_scores[idx];
    
    // Apply causal mask
    if (col > row) {
        attn_scores[idx] = -INFINITY;
    }
}

__global__ void softmax_kernel(
    float* attn_scores,  // Input: scores, Output: probabilities
    float* attn_weights,
    int B, int nh, int T, int batch_idx, int head_idx
) {
    extern __shared__ float shared_data[];
    float* max_val = shared_data;
    float* sum_exp = shared_data + blockDim.x;
    
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    if (row >= T) return;
    
    int idx = row * T + col;
    int global_idx = batch_idx * nh * T * T + head_idx * T * T + idx;
    
    // Thread-local computation for finding max
    if (col == 0) {
        float max_val_local = -INFINITY;
        for (int i = 0; i <= row; i++) {
            float val = attn_scores[batch_idx * nh * T * T + head_idx * T * T + row * T + i];
            max_val_local = fmaxf(max_val_local, val);
        }
        max_val[row] = max_val_local;
        sum_exp[row] = 0.0f;
    }
    __syncthreads();
    
    float local_max = max_val[row];
    
    // Compute exp and sum in a cooperative way
    __shared__ float shared_exp[32*32]; // Temporary storage for exponentials
    if (col <= row) {
        float exp_val = expf(attn_scores[global_idx] - local_max);
        shared_exp[idx] = exp_val;
        atomicAdd(&sum_exp[row], exp_val);
    } else {
        shared_exp[idx] = 0.0f;
    }
    __syncthreads();
    
    // Normalize
    if (col <= row) {
        float sum = sum_exp[row];
        attn_weights[global_idx] = shared_exp[idx] / sum;
    } else {
        attn_weights[global_idx] = 0.0f;
    }
}

torch::Tensor fused_attention_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor bias, float scale
) {
    auto B = q.size(0);
    auto nh = q.size(1);
    auto T = q.size(2);
    auto hs = q.size(3);
    
    // Reshape tensors for batch matrix multiplication
    auto q_flat = q.reshape({B * nh, T, hs});
    auto k_flat = k.reshape({B * nh, T, hs});
    auto v_flat = v.reshape({B * nh, T, hs});
    
    // Step 1: Q @ K^T (highly optimized BLAS operation)
    auto attn_scores = torch::bmm(q_flat, k_flat.transpose(-2, -1));
    attn_scores = attn_scores * scale;
    attn_scores = attn_scores.reshape({B, nh, T, T});
    
    // Step 2: Apply causal mask (custom kernel)
    dim3 block_mask(32, 32);  // 32x32 threads for tile processing
    dim3 grid_mask(B, nh);
    
    masked_softmax_kernel<<<grid_mask, block_mask>>>(
        attn_scores.data_ptr<float>(),
        bias.data_ptr<float>(),
        B, nh, T
    );
    
    // Step 3: Apply softmax (custom kernel with shared memory)
    auto attn_weights = torch::empty_like(attn_scores);
    
    dim3 block_soft(32, 32);  // 32x32 threads
    dim3 grid_soft(B, nh);
    size_t shared_mem_size = 2 * T * sizeof(float); // For max_val and sum_exp
    
    softmax_kernel<<<grid_soft, block_soft, shared_mem_size>>>(
        attn_scores.data_ptr<float>(),
        attn_weights.data_ptr<float>(),
        B, nh, T, 0, 0
    );
    
    // Step 4: attn_weights @ V (highly optimized BLAS operation)
    auto attn_weights_flat = attn_weights.reshape({B * nh, T, T});
    auto out = torch::bmm(attn_weights_flat, v_flat);
    out = out.reshape({B, nh, T, hs});
    
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
        k = k.view(B, T, self.n_head, C // self.n_head).permute(0, 2, 1, 3).contiguous()
        q = q.view(B, T, self.n_head, C // self.n_head).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, T, self.n_head, C // self.n_head).permute(0, 2, 1, 3).contiguous()
        
        # Use fused attention kernel with separated BLAS and custom softmax
        scale = 1.0 / math.sqrt(k.size(-1))
        y = self.fused_attention.fused_attention_forward(
            q, k, v, self.bias[:, :, :T, :T], scale
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
    return [torch.rand(batch_size, seq_len, n_embd, device='cuda')]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]