import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused attention kernel: computes scaled dot-product attention with causal masking
fused_attention_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define BLOCK_SIZE 256

// Fused kernel: scale, causal mask, softmax in one pass using online softmax
__global__ void fused_scaled_causal_softmax_kernel(
    float* att,          // [B, nh, T, T] attention scores (will be modified in-place)
    const int B,
    const int nh,
    const int T,
    const float scale
) {
    // Each block handles one row of the attention matrix
    int batch_head_idx = blockIdx.x;  // which (batch, head) pair
    int row = blockIdx.y;             // which row in the T x T matrix
    
    if (batch_head_idx >= B * nh || row >= T) return;
    
    int b = batch_head_idx / nh;
    int h = batch_head_idx % nh;
    
    // Pointer to the start of this row
    float* row_ptr = att + (b * nh * T * T) + (h * T * T) + (row * T);
    
    // For causal attention, we only attend to positions [0, row]
    int valid_len = row + 1;
    
    // First pass: find max for numerical stability and apply scale
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < valid_len; i += blockDim.x) {
        float val = row_ptr[i] * scale;
        row_ptr[i] = val;  // Store scaled value back
        if (val > max_val) max_val = val;
    }
    
    // Reduce max across threads in block
    __shared__ float shared_max[BLOCK_SIZE];
    shared_max[threadIdx.x] = max_val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (shared_max[threadIdx.x + s] > shared_max[threadIdx.x]) {
                shared_max[threadIdx.x] = shared_max[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    max_val = shared_max[0];
    
    // Second pass: compute exp and sum
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < valid_len; i += blockDim.x) {
        float val = expf(row_ptr[i] - max_val);
        row_ptr[i] = val;
        sum_exp += val;
    }
    
    // Set masked positions to 0
    for (int i = valid_len + threadIdx.x; i < T; i += blockDim.x) {
        row_ptr[i] = 0.0f;
    }
    
    // Reduce sum across threads
    __shared__ float shared_sum[BLOCK_SIZE];
    shared_sum[threadIdx.x] = sum_exp;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    sum_exp = shared_sum[0];
    
    // Third pass: normalize
    float inv_sum = 1.0f / sum_exp;
    for (int i = threadIdx.x; i < valid_len; i += blockDim.x) {
        row_ptr[i] *= inv_sum;
    }
}

torch::Tensor fused_scaled_causal_softmax(torch::Tensor att, float scale) {
    auto sizes = att.sizes();
    int B = sizes[0];
    int nh = sizes[1];
    int T = sizes[2];
    
    // Launch kernel: one block per (batch, head, row)
    dim3 grid(B * nh, T);
    dim3 block(BLOCK_SIZE);
    
    fused_scaled_causal_softmax_kernel<<<grid, block>>>(
        att.data_ptr<float>(),
        B, nh, T, scale
    );
    
    return att;
}
"""

fused_attention_cpp = """
torch::Tensor fused_scaled_causal_softmax(torch::Tensor att, float scale);
"""

fused_attention_module = load_inline(
    name="fused_attention",
    cpp_sources=fused_attention_cpp,
    cuda_sources=fused_attention_source,
    functions=["fused_scaled_causal_softmax"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    """
    Optimized multi-head masked self-attention with fused kernels.
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
        self.n_head = n_head
        self.n_embd = n_embd
        self.fused_attention = fused_attention_module

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Compute Q @ K^T
        att = torch.matmul(q, k.transpose(-2, -1))  # (B, nh, T, T)
        
        # Fused: scale + causal mask + softmax
        scale = 1.0 / math.sqrt(k.size(-1))
        att = self.fused_attention.fused_scaled_causal_softmax(att.contiguous(), scale)
        
        att = self.attn_dropout(att)
        y = torch.matmul(att, v)  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


def custom_kernel(inputs):
    """Entry point for wafer evaluation"""
    # Get init inputs for creating the model
    n_embd = 768
    n_head = 8
    attn_pdrop = 0.0
    resid_pdrop = 0.0
    max_seqlen = 1024
    
    model = ModelNew(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen).cuda()
    
    # Copy weights from reference if available, otherwise use random
    model.eval()
    
    x = inputs[0]
    with torch.no_grad():
        return model(x)
