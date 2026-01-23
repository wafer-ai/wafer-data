import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused attention kernel using online softmax (Flash Attention style)
fused_attention_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define BLOCK_SIZE 64
#define WARP_SIZE 64

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Fused scaled dot-product attention with causal mask
// Each block handles one row of the attention matrix
__global__ void fused_attention_kernel(
    const float* __restrict__ Q,  // [B, nh, T, hs]
    const float* __restrict__ K,  // [B, nh, T, hs]
    const float* __restrict__ V,  // [B, nh, T, hs]
    float* __restrict__ out,       // [B, nh, T, hs]
    int B, int nh, int T, int hs,
    float scale
) {
    int batch_head_idx = blockIdx.x;  // B * nh
    int row = blockIdx.y;             // which row of the T x T attention matrix
    int tid = threadIdx.x;
    
    int b = batch_head_idx / nh;
    int h = batch_head_idx % nh;
    
    if (b >= B || row >= T) return;
    
    // Pointer to this batch/head
    const float* q_ptr = Q + (b * nh * T * hs) + (h * T * hs) + (row * hs);
    const float* k_base = K + (b * nh * T * hs) + (h * T * hs);
    const float* v_base = V + (b * nh * T * hs) + (h * T * hs);
    float* out_ptr = out + (b * nh * T * hs) + (h * T * hs) + (row * hs);
    
    // Load Q row into registers/shared memory
    extern __shared__ float smem[];
    float* q_shared = smem;
    float* kv_shared = smem + hs;
    
    // Load q row
    for (int i = tid; i < hs; i += blockDim.x) {
        q_shared[i] = q_ptr[i];
    }
    __syncthreads();
    
    // Online softmax variables
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    
    // Accumulator for output (one per thread for its portion of hs)
    float acc[8] = {0.0f};  // Assuming hs <= 8 * blockDim.x
    
    // Process columns up to row (causal mask)
    for (int col = 0; col <= row; col++) {
        // Compute dot product Q[row] @ K[col]
        float dot = 0.0f;
        const float* k_ptr = k_base + col * hs;
        
        for (int i = tid; i < hs; i += blockDim.x) {
            dot += q_shared[i] * k_ptr[i];
        }
        
        // Reduce within warp
        dot = warp_reduce_sum(dot);
        dot *= scale;
        
        // Online softmax update (only thread 0 computes, then broadcasts)
        float old_max = max_val;
        max_val = fmaxf(max_val, dot);
        float exp_diff = expf(old_max - max_val);
        sum_exp = sum_exp * exp_diff + expf(dot - max_val);
        
        // Load V[col] and accumulate weighted
        const float* v_ptr = v_base + col * hs;
        float weight = expf(dot - max_val);
        
        for (int i = 0; i < 8 && tid + i * blockDim.x < hs; i++) {
            int idx = tid + i * blockDim.x;
            acc[i] = acc[i] * exp_diff + weight * v_ptr[idx];
        }
    }
    
    // Normalize and write output
    float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < 8 && tid + i * blockDim.x < hs; i++) {
        int idx = tid + i * blockDim.x;
        out_ptr[idx] = acc[i] * inv_sum;
    }
}

torch::Tensor fused_attention_hip(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // Q, K, V: [B, nh, T, hs]
    auto B = Q.size(0);
    auto nh = Q.size(1);
    auto T = Q.size(2);
    auto hs = Q.size(3);
    
    auto out = torch::empty_like(Q);
    float scale = 1.0f / sqrtf((float)hs);
    
    dim3 grid(B * nh, T);
    int block_size = min(64, (int)hs);
    int smem_size = hs * sizeof(float) * 2;
    
    fused_attention_kernel<<<grid, block_size, smem_size>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        out.data_ptr<float>(),
        B, nh, T, hs,
        scale
    );
    
    return out;
}
"""

cpp_source = """
torch::Tensor fused_attention_hip(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
"""

fused_attention = load_inline(
    name="fused_attention",
    cpp_sources=cpp_source,
    cuda_sources=fused_attention_source,
    functions=["fused_attention_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


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
        self.fused_attention = fused_attention

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Make tensors contiguous for the fused kernel
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Use fused attention kernel
        y = self.fused_attention.fused_attention_hip(q, k, v)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


def get_inputs():
    return [torch.rand(128, 512, 768).cuda()]


def get_init_inputs():
    return [768, 8, 0.0, 0.0, 1024]
