
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

fused_layernorm_residual_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 32; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset, 64);
    }
    return val;
}

__global__ void fused_layernorm_residual_kernel(
    const float* __restrict__ attn_output, // (B, L, C)
    const float* __restrict__ x_original,  // (B, C, H, W)
    const float* __restrict__ gamma,       // (C)
    const float* __restrict__ beta,        // (C)
    float* __restrict__ output,            // (B, C, H, W)
    int B, int L, int C, int H, int W, float eps) {

    int b = blockIdx.x;
    int l = blockIdx.y;
    int tid = threadIdx.x;

    if (b >= B || l >= L) return;

    // We assume C=128 and blockDim.x=128
    float val = 0.0f;
    if (tid < C) {
        int idx_attn = (b * L + l) * C + tid;
        int h = l / W;
        int w = l % W;
        int idx_orig = ((b * C + tid) * H + h) * W + w;
        val = attn_output[idx_attn] + x_original[idx_orig];
    }

    // Shared memory for reduction
    __shared__ float shared_sum[4]; // 128 threads = 2 warps of 64 on MI300X or 4 warps of 32
    
    // warp_size is 64 on MI300X/CDNA
    float sum = val;
    for (int offset = 32; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset, 64);
    }
    
    int warp_id = tid / 64;
    int lane_id = tid % 64;
    if (lane_id == 0) shared_sum[warp_id] = sum;
    __syncthreads();
    
    float total_sum = (tid < 2) ? shared_sum[tid] : 0.0f; // MI300X has warp size 64
    if (tid < 64) {
        for (int offset = 1; offset > 0; offset >>= 1) { // only 2 warps
           total_sum += __shfl_xor(total_sum, offset, 64);
        }
    }
    
    __shared__ float mean_shared;
    if (tid == 0) mean_shared = total_sum / C;
    __syncthreads();
    float mean = mean_shared;

    float diff = (tid < C) ? (val - mean) : 0.0f;
    float sum_sq = diff * diff;
    for (int offset = 32; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor(sum_sq, offset, 64);
    }
    if (lane_id == 0) shared_sum[warp_id] = sum_sq;
    __syncthreads();
    
    float total_sum_sq = (tid < 2) ? shared_sum[tid] : 0.0f;
    if (tid < 64) {
        for (int offset = 1; offset > 0; offset >>= 1) {
           total_sum_sq += __shfl_xor(total_sum_sq, offset, 64);
        }
    }
    
    __shared__ float inv_std_shared;
    if (tid == 0) inv_std_shared = 1.0f / sqrtf(total_sum_sq / C + eps);
    __syncthreads();
    float inv_std = inv_std_shared;

    if (tid < C) {
        float out_val = diff * inv_std * gamma[tid] + beta[tid];
        int h = l / W;
        int w = l % W;
        int out_idx = ((b * C + tid) * H + h) * W + w;
        output[out_idx] = out_val;
    }
}

torch::Tensor fused_layernorm_residual(
    torch::Tensor attn_output,
    torch::Tensor x_original,
    torch::Tensor gamma,
    torch::Tensor beta,
    int H, int W, float eps) {
    
    int B = attn_output.size(0);
    int L = attn_output.size(1);
    int C = attn_output.size(2);

    auto output = torch::empty({B, C, H, W}, attn_output.options());

    dim3 grid(B, L);
    int block_size = 128; 

    fused_layernorm_residual_kernel<<<grid, block_size>>>(
        attn_output.data_ptr<float>(),
        x_original.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        B, L, C, H, W, eps);

    return output;
}
"""

fused_ops = load_inline(
    name="fused_ops_v5",
    cpp_sources=fused_layernorm_residual_cpp_source,
    functions=["fused_layernorm_residual"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(ModelNew, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W
        
        # QKV projection
        x_flat = x.view(B, C, L).transpose(1, 2) # (B, L, C)
        qkv = torch.addmm(self.attn.in_proj_bias, x_flat.reshape(-1, C), self.attn.in_proj_weight.t())
        qkv = qkv.view(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot product attention
        attn_output = F.scaled_dot_product_attention(q, k, v)
        
        # Combined heads and output projection
        attn_output = attn_output.transpose(1, 2).reshape(B, L, self.embed_dim)
        attn_output = torch.addmm(self.attn.out_proj.bias, attn_output.reshape(-1, self.embed_dim), self.attn.out_proj.weight.t())
        attn_output = attn_output.view(B, L, self.embed_dim)
        
        # Fused residual, layer norm, and permute back
        x = fused_ops.fused_layernorm_residual(
            attn_output,
            x,
            self.norm.weight,
            self.norm.bias,
            H, W,
            self.norm.eps
        )
        return x

def get_inputs():
    return [torch.rand(2, 128, 128, 128).cuda()]

def get_init_inputs():
    return [128, 4]
