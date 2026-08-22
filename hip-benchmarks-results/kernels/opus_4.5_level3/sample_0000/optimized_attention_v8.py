import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

os.environ["CXX"] = "hipcc"

# Optimized fused kernels
fused_kernels_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define WARP_SIZE 64

// Fused residual add + LayerNorm kernel with vectorized loads
// Dynamic block size version
__global__ void fused_residual_layernorm_vec4_kernel(
    const float4* __restrict__ attn_out,
    const float4* __restrict__ residual,
    const float4* __restrict__ weight,
    const float4* __restrict__ bias,
    float4* __restrict__ out,
    int num_elements,
    int embed_dim_vec4,
    float eps,
    int block_size
) {
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    
    int base_offset = idx * embed_dim_vec4;
    
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    for (int i = tid; i < embed_dim_vec4; i += block_size) {
        float4 a = attn_out[base_offset + i];
        float4 r = residual[base_offset + i];
        
        float v0 = a.x + r.x;
        float v1 = a.y + r.y;
        float v2 = a.z + r.z;
        float v3 = a.w + r.w;
        
        local_sum += v0 + v1 + v2 + v3;
        local_sum_sq += v0*v0 + v1*v1 + v2*v2 + v3*v3;
    }
    
    // Warp-level reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down(local_sum, offset, WARP_SIZE);
        local_sum_sq += __shfl_down(local_sum_sq, offset, WARP_SIZE);
    }
    
    // Dynamic shared memory
    extern __shared__ float shared_mem[];
    float* s_sum = shared_mem;
    float* s_sum_sq = shared_mem + (block_size / WARP_SIZE);
    
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = block_size / WARP_SIZE;
    
    if (lane == 0) {
        s_sum[warp_id] = local_sum;
        s_sum_sq[warp_id] = local_sum_sq;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        local_sum = (lane < num_warps) ? s_sum[lane] : 0.0f;
        local_sum_sq = (lane < num_warps) ? s_sum_sq[lane] : 0.0f;
        
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            local_sum += __shfl_down(local_sum, offset, WARP_SIZE);
            local_sum_sq += __shfl_down(local_sum_sq, offset, WARP_SIZE);
        }
    }
    
    __shared__ float mean, inv_std;
    if (tid == 0) {
        int embed_dim = embed_dim_vec4 * 4;
        mean = local_sum / embed_dim;
        float variance = local_sum_sq / embed_dim - mean * mean;
        inv_std = rsqrtf(variance + eps);
    }
    __syncthreads();
    
    for (int i = tid; i < embed_dim_vec4; i += block_size) {
        float4 a = attn_out[base_offset + i];
        float4 r = residual[base_offset + i];
        float4 w = weight[i];
        float4 b = bias[i];
        
        float4 result;
        result.x = ((a.x + r.x) - mean) * inv_std * w.x + b.x;
        result.y = ((a.y + r.y) - mean) * inv_std * w.y + b.y;
        result.z = ((a.z + r.z) - mean) * inv_std * w.z + b.z;
        result.w = ((a.w + r.w) - mean) * inv_std * w.w + b.w;
        
        out[base_offset + i] = result;
    }
}

torch::Tensor fused_residual_layernorm_hip(
    torch::Tensor attn_out,
    torch::Tensor residual,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
) {
    int seq_len = attn_out.size(0);
    int batch_size = attn_out.size(1);
    int embed_dim = attn_out.size(2);
    
    auto out = torch::empty_like(attn_out);
    
    int num_blocks = seq_len * batch_size;
    // Use appropriate block size (must be multiple of WARP_SIZE=64)
    int block_size = 256;
    
    int shared_mem_size = 2 * (block_size / WARP_SIZE) * sizeof(float);
    
    fused_residual_layernorm_vec4_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        reinterpret_cast<const float4*>(attn_out.data_ptr<float>()),
        reinterpret_cast<const float4*>(residual.data_ptr<float>()),
        reinterpret_cast<const float4*>(weight.data_ptr<float>()),
        reinterpret_cast<const float4*>(bias.data_ptr<float>()),
        reinterpret_cast<float4*>(out.data_ptr<float>()),
        num_blocks,
        embed_dim / 4,
        eps,
        block_size
    );
    
    return out;
}
"""

fused_kernels_cpp = """
torch::Tensor fused_residual_layernorm_hip(
    torch::Tensor attn_out,
    torch::Tensor residual,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
);
"""

fused_module = load_inline(
    name="fused_attention_kernels_v8",
    cpp_sources=fused_kernels_cpp,
    cuda_sources=fused_kernels_source,
    functions=["fused_residual_layernorm_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"],
)


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Attention Block using optimized scaled_dot_product_attention.
        :param embed_dim: Embedding dimension (the number of channels)
        :param num_heads: Number of attention heads
        """
        super(ModelNew, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Use nn.MultiheadAttention's weights for compatibility
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.fused_module = fused_module
        self.eps = self.norm.eps
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x):
        """
        Forward pass of the AttentionBlock.
        :param x: Input tensor of shape (B, C, H, W)
        :return: Output tensor of the same shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        seq_len = H * W
        
        # Reshape: (B, C, H, W) -> (seq_len, B, C)
        x_seq = x.view(B, C, seq_len).permute(2, 0, 1).contiguous()
        
        # Extract weights
        in_proj_weight = self.attn.in_proj_weight
        in_proj_bias = self.attn.in_proj_bias
        out_proj_weight = self.attn.out_proj.weight
        out_proj_bias = self.attn.out_proj.bias
        
        # Compute Q, K, V using in_proj
        qkv = F.linear(x_seq, in_proj_weight, in_proj_bias)  # (seq_len, B, 3*C)
        qkv = qkv.view(seq_len, B, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 1, 3, 0, 4)  # (3, B, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, seq_len, head_dim)
        
        # Use PyTorch's optimized scaled_dot_product_attention
        attn_output = F.scaled_dot_product_attention(q, k, v, scale=self.scale)  # (B, num_heads, seq_len, head_dim)
        
        # Reshape: (B, num_heads, seq_len, head_dim) -> (seq_len, B, C)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(seq_len, B, C)
        
        # Output projection
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
        
        # Fused residual add + LayerNorm
        x_norm = self.fused_module.fused_residual_layernorm_hip(
            attn_output.contiguous(), 
            x_seq,
            self.norm.weight,
            self.norm.bias,
            self.eps
        )
        
        # Reshape back: (seq_len, B, C) -> (B, C, H, W)
        out = x_norm.permute(1, 2, 0).view(B, C, H, W)
        
        return out
