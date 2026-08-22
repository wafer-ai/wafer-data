import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused softmax + masking kernel
fused_softmax_mask_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>
#include <float.h>

#define WARP_SIZE 64

// Warp-level reduction for max
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

// Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Block-level reduction for max
__device__ float block_reduce_max(float val, float* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_max(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : -FLT_MAX;
    if (wid == 0) val = warp_reduce_max(val);
    
    return val;
}

// Block-level reduction for sum
__device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    
    return val;
}

// Fused scale, causal mask, and softmax kernel
// att: [B*nh, T, T] - each block handles one row
__global__ void fused_softmax_mask_kernel(
    float* __restrict__ att,
    int T,
    float scale
) {
    int batch_head = blockIdx.x;  // B * nh
    int row = blockIdx.y;         // row index
    
    float* att_row = att + batch_head * T * T + row * T;
    
    extern __shared__ float shared[];
    
    // Step 1: Apply scale and mask, find max
    float local_max = -FLT_MAX;
    for (int col = threadIdx.x; col < T; col += blockDim.x) {
        float val;
        if (col <= row) {
            val = att_row[col] * scale;
        } else {
            val = -FLT_MAX;
        }
        att_row[col] = val;
        local_max = fmaxf(local_max, val);
    }
    
    // Reduce max across block
    __shared__ float smem[32];
    float max_val = block_reduce_max(local_max, smem);
    __syncthreads();
    
    // Broadcast max_val
    if (threadIdx.x == 0) shared[0] = max_val;
    __syncthreads();
    max_val = shared[0];
    
    // Step 2: Compute exp and sum
    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < T; col += blockDim.x) {
        float val = att_row[col];
        float exp_val = (val > -FLT_MAX / 2.0f) ? expf(val - max_val) : 0.0f;
        att_row[col] = exp_val;
        local_sum += exp_val;
    }
    
    // Reduce sum across block
    float sum_val = block_reduce_sum(local_sum, smem);
    __syncthreads();
    
    // Broadcast sum_val
    if (threadIdx.x == 0) shared[0] = sum_val;
    __syncthreads();
    sum_val = shared[0];
    
    // Step 3: Normalize
    float inv_sum = 1.0f / (sum_val + 1e-9f);
    for (int col = threadIdx.x; col < T; col += blockDim.x) {
        att_row[col] *= inv_sum;
    }
}

void fused_softmax_mask_hip(torch::Tensor att, int T, float scale) {
    int BNH = att.size(0);
    
    dim3 grid(BNH, T);
    int block_size = 256;
    int smem_size = sizeof(float) * 33;  // For reductions
    
    fused_softmax_mask_kernel<<<grid, block_size, smem_size>>>(
        att.data_ptr<float>(),
        T,
        scale
    );
}
"""

cpp_source = """
void fused_softmax_mask_hip(torch::Tensor att, int T, float scale);
"""

fused_softmax_mask = load_inline(
    name="fused_softmax_mask",
    cpp_sources=cpp_source,
    cuda_sources=fused_softmax_mask_source,
    functions=["fused_softmax_mask_hip"],
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
        self.fused_softmax_mask = fused_softmax_mask

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Compute Q @ K^T
        att = torch.matmul(q, k.transpose(-2, -1))  # [B, nh, T, T]
        
        # Reshape for kernel: [B*nh, T, T]
        att = att.view(B * self.n_head, T, T).contiguous()
        
        # Fused scale + mask + softmax
        scale = 1.0 / math.sqrt(k.size(-1))
        self.fused_softmax_mask.fused_softmax_mask_hip(att, T, scale)
        
        # Reshape back: [B, nh, T, T]
        att = att.view(B, self.n_head, T, T)
        
        att = self.attn_dropout(att)
        y = torch.matmul(att, v)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


def get_inputs():
    return [torch.rand(128, 512, 768).cuda()]


def get_init_inputs():
    return [768, 8, 0.0, 0.0, 1024]
