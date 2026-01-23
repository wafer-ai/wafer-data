import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized fused softmax + masking kernel
# Uses vectorized loads and better memory coalescing
fused_softmax_mask_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>
#include <float.h>

#define WARP_SIZE 64
#define BLOCK_SIZE 256

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

// Optimized fused scale, causal mask, and softmax kernel
// att: [B*nh, T, T] - each block handles one row
__global__ void fused_softmax_mask_kernel(
    float* __restrict__ att,
    const int T,
    const float scale
) {
    const int batch_head = blockIdx.x;  // B * nh
    const int row = blockIdx.y;         // row index
    
    float* att_row = att + batch_head * T * T + row * T;
    
    __shared__ float shared_max;
    __shared__ float shared_sum;
    
    // Step 1: Apply scale and mask, find local max using vectorized loads
    float local_max = -FLT_MAX;
    
    // Process 4 elements at a time when possible
    int num_vec = (row + 1) / 4;  // Number of complete float4 vectors in valid region
    int remaining = (row + 1) % 4;
    
    // Vectorized processing
    float4* att_row_vec = reinterpret_cast<float4*>(att_row);
    for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
        float4 v = att_row_vec[i];
        v.x *= scale;
        v.y *= scale;
        v.z *= scale;
        v.w *= scale;
        att_row_vec[i] = v;
        local_max = fmaxf(local_max, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    }
    
    // Process remaining valid elements
    for (int col = num_vec * 4 + threadIdx.x; col <= row; col += blockDim.x) {
        float val = att_row[col] * scale;
        att_row[col] = val;
        local_max = fmaxf(local_max, val);
    }
    
    // Set masked elements to -inf
    for (int col = row + 1 + threadIdx.x; col < T; col += blockDim.x) {
        att_row[col] = -FLT_MAX;
    }
    
    // Reduce max across block using warp shuffles
    local_max = warp_reduce_max(local_max);
    
    // Warp leaders write to shared memory
    __shared__ float warp_max[BLOCK_SIZE / WARP_SIZE];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    
    if (lane == 0) warp_max[warp_id] = local_max;
    __syncthreads();
    
    // First warp reduces the warp maxes
    float max_val = -FLT_MAX;
    if (threadIdx.x < BLOCK_SIZE / WARP_SIZE) {
        max_val = warp_max[threadIdx.x];
    }
    max_val = warp_reduce_max(max_val);
    
    if (threadIdx.x == 0) shared_max = max_val;
    __syncthreads();
    max_val = shared_max;
    
    // Step 2: Compute exp and sum
    float local_sum = 0.0f;
    
    // Vectorized exp computation
    for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
        float4 v = att_row_vec[i];
        v.x = expf(v.x - max_val);
        v.y = expf(v.y - max_val);
        v.z = expf(v.z - max_val);
        v.w = expf(v.w - max_val);
        att_row_vec[i] = v;
        local_sum += v.x + v.y + v.z + v.w;
    }
    
    // Process remaining
    for (int col = num_vec * 4 + threadIdx.x; col <= row; col += blockDim.x) {
        float val = expf(att_row[col] - max_val);
        att_row[col] = val;
        local_sum += val;
    }
    
    // Masked elements get 0
    for (int col = row + 1 + threadIdx.x; col < T; col += blockDim.x) {
        att_row[col] = 0.0f;
    }
    
    // Reduce sum across block
    local_sum = warp_reduce_sum(local_sum);
    
    __shared__ float warp_sum[BLOCK_SIZE / WARP_SIZE];
    if (lane == 0) warp_sum[warp_id] = local_sum;
    __syncthreads();
    
    float sum_val = 0.0f;
    if (threadIdx.x < BLOCK_SIZE / WARP_SIZE) {
        sum_val = warp_sum[threadIdx.x];
    }
    sum_val = warp_reduce_sum(sum_val);
    
    if (threadIdx.x == 0) shared_sum = sum_val;
    __syncthreads();
    sum_val = shared_sum;
    
    // Step 3: Normalize
    float inv_sum = 1.0f / (sum_val + 1e-9f);
    
    // Vectorized normalization
    for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
        float4 v = att_row_vec[i];
        v.x *= inv_sum;
        v.y *= inv_sum;
        v.z *= inv_sum;
        v.w *= inv_sum;
        att_row_vec[i] = v;
    }
    
    // Remaining
    for (int col = num_vec * 4 + threadIdx.x; col <= row; col += blockDim.x) {
        att_row[col] *= inv_sum;
    }
}

void fused_softmax_mask_hip(torch::Tensor att, int T, float scale) {
    int BNH = att.size(0);
    
    dim3 grid(BNH, T);
    int block_size = BLOCK_SIZE;
    
    fused_softmax_mask_kernel<<<grid, block_size>>>(
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
    name="fused_softmax_mask_v3",
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

        # Compute Q @ K^T using efficient batched matmul
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
