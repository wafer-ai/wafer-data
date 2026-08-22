import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused attention kernel with better memory access patterns
fused_attention_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define WARP_SIZE 64
#define BLOCK_DIM 256

// Warp reduce max
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

// Warp reduce sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Fused kernel: scale, causal mask, softmax using warp-level primitives
__global__ void fused_scaled_causal_softmax_kernel(
    float* __restrict__ att,
    const int B,
    const int nh,
    const int T,
    const float scale
) {
    // Each warp handles one row
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int total_rows = B * nh * T;
    
    if (warp_id >= total_rows) return;
    
    // Decode which row we're processing
    const int row_in_head = warp_id % T;
    const int head_batch_idx = warp_id / T;
    
    // Pointer to the start of this row
    float* row_ptr = att + warp_id * T;
    
    // For causal attention, we only attend to positions [0, row_in_head]
    const int valid_len = row_in_head + 1;
    
    // Pass 1: find max for numerical stability
    float max_val = -INFINITY;
    for (int i = lane_id; i < valid_len; i += WARP_SIZE) {
        float val = row_ptr[i] * scale;
        row_ptr[i] = val;
        max_val = fmaxf(max_val, val);
    }
    max_val = warp_reduce_max(max_val);
    
    // Pass 2: compute exp and sum
    float sum_exp = 0.0f;
    for (int i = lane_id; i < valid_len; i += WARP_SIZE) {
        float val = expf(row_ptr[i] - max_val);
        row_ptr[i] = val;
        sum_exp += val;
    }
    sum_exp = warp_reduce_sum(sum_exp);
    
    // Pass 3: normalize and set masked positions to 0
    float inv_sum = 1.0f / sum_exp;
    for (int i = lane_id; i < valid_len; i += WARP_SIZE) {
        row_ptr[i] *= inv_sum;
    }
    for (int i = valid_len + lane_id; i < T; i += WARP_SIZE) {
        row_ptr[i] = 0.0f;
    }
}

torch::Tensor fused_scaled_causal_softmax(torch::Tensor att, float scale) {
    auto sizes = att.sizes();
    int B = sizes[0];
    int nh = sizes[1];
    int T = sizes[2];
    
    int total_rows = B * nh * T;
    int warps_per_block = BLOCK_DIM / WARP_SIZE;
    int num_blocks = (total_rows + warps_per_block - 1) / warps_per_block;
    
    fused_scaled_causal_softmax_kernel<<<num_blocks, BLOCK_DIM>>>(
        att.data_ptr<float>(),
        B, nh, T, scale
    );
    
    return att;
}

// Fused QK^T with scale kernel for better memory efficiency
__global__ void fused_qk_scale_kernel(
    const float* __restrict__ q,    // [B, nh, T, hs]
    const float* __restrict__ k,    // [B, nh, T, hs]
    float* __restrict__ att,        // [B, nh, T, T]
    const int B,
    const int nh,
    const int T,
    const int hs,
    const float scale
) {
    // Each block computes a tile of the output
    const int b_h = blockIdx.z;  // batch and head combined
    const int row = blockIdx.y * blockDim.y + threadIdx.y;  // query position
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  // key position
    
    if (row >= T || col >= T) return;
    
    const int b = b_h / nh;
    const int h = b_h % nh;
    
    // Compute dot product
    const float* q_row = q + b * nh * T * hs + h * T * hs + row * hs;
    const float* k_row = k + b * nh * T * hs + h * T * hs + col * hs;
    
    float sum = 0.0f;
    for (int i = 0; i < hs; i++) {
        sum += q_row[i] * k_row[i];
    }
    
    // Apply scale and causal mask
    if (col > row) {
        sum = -INFINITY;
    } else {
        sum *= scale;
    }
    
    att[b * nh * T * T + h * T * T + row * T + col] = sum;
}

torch::Tensor fused_qk_scale(torch::Tensor q, torch::Tensor k, float scale) {
    auto sizes = q.sizes();
    int B = sizes[0];
    int nh = sizes[1];
    int T = sizes[2];
    int hs = sizes[3];
    
    auto att = torch::empty({B, nh, T, T}, q.options());
    
    dim3 block(16, 16);
    dim3 grid((T + 15) / 16, (T + 15) / 16, B * nh);
    
    fused_qk_scale_kernel<<<grid, block>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        att.data_ptr<float>(),
        B, nh, T, hs, scale
    );
    
    return att;
}
"""

fused_attention_cpp = """
torch::Tensor fused_scaled_causal_softmax(torch::Tensor att, float scale);
torch::Tensor fused_qk_scale(torch::Tensor q, torch::Tensor k, float scale);
"""

fused_attention_module = load_inline(
    name="fused_attention_v2",
    cpp_sources=fused_attention_cpp,
    cuda_sources=fused_attention_source,
    functions=["fused_scaled_causal_softmax", "fused_qk_scale"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized multi-head masked self-attention with fused kernels.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.n_embd = n_embd
        self.fused_attention = fused_attention_module

    def forward(self, x):
        B, T, C = x.size()
        hs = C // self.n_head

        # calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2).contiguous()  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2).contiguous()  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)

        # Compute Q @ K^T with scale
        scale = 1.0 / math.sqrt(hs)
        att = torch.matmul(q, k.transpose(-2, -1))  # (B, nh, T, T)
        
        # Fused: scale + causal mask + softmax
        att = self.fused_attention.fused_scaled_causal_softmax(att.contiguous(), scale)
        
        att = self.attn_dropout(att)
        y = torch.matmul(att, v)  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
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
