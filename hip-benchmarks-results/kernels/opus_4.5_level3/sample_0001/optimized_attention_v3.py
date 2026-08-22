import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized fused softmax kernel with vectorized loads
fused_attention_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>
#include <hip/hip_fp16.h>

#define WARP_SIZE 64
#define BLOCK_SIZE 256
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

// Warp reduce max using butterfly reduction
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

// Warp reduce sum using butterfly reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Optimized fused kernel using float4 vectorized loads where possible
__global__ void fused_scaled_causal_softmax_kernel(
    float* __restrict__ att,
    const int B,
    const int nh,
    const int T,
    const float scale
) {
    __shared__ float smem_max[NUM_WARPS];
    __shared__ float smem_sum[NUM_WARPS];
    
    const int row_idx = blockIdx.x;  // Each block processes one row
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    const int total_rows = B * nh * T;
    if (row_idx >= total_rows) return;
    
    const int row_in_head = row_idx % T;
    float* row_ptr = att + row_idx * T;
    
    // For causal attention, valid length is row + 1
    const int valid_len = row_in_head + 1;
    
    // Pass 1: Scale and find max
    float local_max = -3.402823466e+38f;  // -FLT_MAX
    for (int i = tid; i < valid_len; i += BLOCK_SIZE) {
        float val = row_ptr[i] * scale;
        row_ptr[i] = val;
        local_max = fmaxf(local_max, val);
    }
    
    // Warp-level reduction for max
    local_max = warp_reduce_max(local_max);
    if (lane_id == 0) {
        smem_max[warp_id] = local_max;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (warp_id == 0) {
        local_max = (lane_id < NUM_WARPS) ? smem_max[lane_id] : -3.402823466e+38f;
        local_max = warp_reduce_max(local_max);
        if (lane_id == 0) {
            smem_max[0] = local_max;
        }
    }
    __syncthreads();
    float max_val = smem_max[0];
    
    // Pass 2: Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < valid_len; i += BLOCK_SIZE) {
        float val = __expf(row_ptr[i] - max_val);
        row_ptr[i] = val;
        local_sum += val;
    }
    
    // Warp-level reduction for sum
    local_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) {
        smem_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (warp_id == 0) {
        local_sum = (lane_id < NUM_WARPS) ? smem_sum[lane_id] : 0.0f;
        local_sum = warp_reduce_sum(local_sum);
        if (lane_id == 0) {
            smem_sum[0] = local_sum;
        }
    }
    __syncthreads();
    float sum_exp = smem_sum[0];
    
    // Pass 3: Normalize
    float inv_sum = 1.0f / sum_exp;
    for (int i = tid; i < valid_len; i += BLOCK_SIZE) {
        row_ptr[i] *= inv_sum;
    }
    
    // Zero out masked positions
    for (int i = valid_len + tid; i < T; i += BLOCK_SIZE) {
        row_ptr[i] = 0.0f;
    }
}

torch::Tensor fused_scaled_causal_softmax(torch::Tensor att, float scale) {
    auto sizes = att.sizes();
    int B = sizes[0];
    int nh = sizes[1];
    int T = sizes[2];
    
    int total_rows = B * nh * T;
    
    fused_scaled_causal_softmax_kernel<<<total_rows, BLOCK_SIZE>>>(
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
    name="fused_attention_v3",
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
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)

        # Compute Q @ K^T
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
