import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused residual add + layer norm kernel with vectorized loads
fused_residual_layernorm_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

// Warp reduce sum using warp shuffle
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Optimized fused residual add + layer norm kernel
// Uses vectorized float4 loads for better memory bandwidth
__global__ void fused_residual_layernorm_kernel_v2(
    const float4* __restrict__ attn_output,
    const float4* __restrict__ residual,
    const float4* __restrict__ gamma,
    const float4* __restrict__ beta,
    float4* __restrict__ output,
    int seq_len,
    int batch_size,
    int embed_dim,
    float eps
) {
    __shared__ float shared_sum[64];
    __shared__ float shared_mean;
    __shared__ float shared_inv_std;
    
    int seq_batch_idx = blockIdx.x;
    int seq_idx = seq_batch_idx / batch_size;
    int batch_idx = seq_batch_idx % batch_size;
    
    if (seq_idx >= seq_len) return;
    
    int embed_dim4 = embed_dim / 4;
    int base_idx = seq_idx * batch_size * embed_dim4 + batch_idx * embed_dim4;
    
    int tid = threadIdx.x;
    int warp_id = tid / 64;
    int lane_id = tid % 64;
    
    // Step 1: Load data and compute partial sums
    float local_sum = 0.0f;
    float local_vals[8];  // Store local values for reuse
    int num_vec4_per_thread = (embed_dim4 + blockDim.x - 1) / blockDim.x;
    
    #pragma unroll 4
    for (int i = 0; i < num_vec4_per_thread; i++) {
        int idx = tid + i * blockDim.x;
        if (idx < embed_dim4) {
            float4 a = attn_output[base_idx + idx];
            float4 r = residual[base_idx + idx];
            float4 val;
            val.x = a.x + r.x;
            val.y = a.y + r.y;
            val.z = a.z + r.z;
            val.w = a.w + r.w;
            
            local_sum += val.x + val.y + val.z + val.w;
        }
    }
    
    // Warp reduce
    local_sum = warp_reduce_sum(local_sum);
    
    if (lane_id == 0) {
        shared_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (tid < 64) {
        float val = (tid < (blockDim.x / 64)) ? shared_sum[tid] : 0.0f;
        val = warp_reduce_sum(val);
        if (tid == 0) {
            shared_mean = val / embed_dim;
        }
    }
    __syncthreads();
    
    float mean = shared_mean;
    
    // Step 2: Compute variance
    float var_sum = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < num_vec4_per_thread; i++) {
        int idx = tid + i * blockDim.x;
        if (idx < embed_dim4) {
            float4 a = attn_output[base_idx + idx];
            float4 r = residual[base_idx + idx];
            float4 val;
            val.x = a.x + r.x;
            val.y = a.y + r.y;
            val.z = a.z + r.z;
            val.w = a.w + r.w;
            
            float d0 = val.x - mean;
            float d1 = val.y - mean;
            float d2 = val.z - mean;
            float d3 = val.w - mean;
            var_sum += d0*d0 + d1*d1 + d2*d2 + d3*d3;
        }
    }
    
    var_sum = warp_reduce_sum(var_sum);
    
    if (lane_id == 0) {
        shared_sum[warp_id] = var_sum;
    }
    __syncthreads();
    
    if (tid < 64) {
        float val = (tid < (blockDim.x / 64)) ? shared_sum[tid] : 0.0f;
        val = warp_reduce_sum(val);
        if (tid == 0) {
            shared_inv_std = rsqrtf(val / embed_dim + eps);
        }
    }
    __syncthreads();
    
    float inv_std = shared_inv_std;
    
    // Step 3: Normalize and write output
    #pragma unroll 4
    for (int i = 0; i < num_vec4_per_thread; i++) {
        int idx = tid + i * blockDim.x;
        if (idx < embed_dim4) {
            float4 a = attn_output[base_idx + idx];
            float4 r = residual[base_idx + idx];
            float4 g = gamma[idx];
            float4 b = beta[idx];
            
            float4 val;
            val.x = a.x + r.x;
            val.y = a.y + r.y;
            val.z = a.z + r.z;
            val.w = a.w + r.w;
            
            float4 out;
            out.x = ((val.x - mean) * inv_std) * g.x + b.x;
            out.y = ((val.y - mean) * inv_std) * g.y + b.y;
            out.z = ((val.z - mean) * inv_std) * g.z + b.z;
            out.w = ((val.w - mean) * inv_std) * g.w + b.w;
            
            output[base_idx + idx] = out;
        }
    }
}

torch::Tensor fused_residual_layernorm_hip(
    torch::Tensor attn_output,
    torch::Tensor residual,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
) {
    auto seq_len = attn_output.size(0);
    auto batch_size = attn_output.size(1);
    auto embed_dim = attn_output.size(2);
    
    auto output = torch::empty_like(attn_output);
    
    int num_blocks = seq_len * batch_size;
    int block_size = 256;
    
    fused_residual_layernorm_kernel_v2<<<num_blocks, block_size>>>(
        reinterpret_cast<const float4*>(attn_output.data_ptr<float>()),
        reinterpret_cast<const float4*>(residual.data_ptr<float>()),
        reinterpret_cast<const float4*>(gamma.data_ptr<float>()),
        reinterpret_cast<const float4*>(beta.data_ptr<float>()),
        reinterpret_cast<float4*>(output.data_ptr<float>()),
        seq_len,
        batch_size,
        embed_dim,
        eps
    );
    
    return output;
}
"""

fused_residual_layernorm_cpp = """
torch::Tensor fused_residual_layernorm_hip(
    torch::Tensor attn_output,
    torch::Tensor residual,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
);
"""

fused_module = load_inline(
    name="fused_residual_layernorm_v2",
    cpp_sources=fused_residual_layernorm_cpp,
    cuda_sources=fused_residual_layernorm_source,
    functions=["fused_residual_layernorm_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Attention Block using Multihead Self-Attention with fused operations.
        :param embed_dim: Embedding dimension (the number of channels)
        :param num_heads: Number of attention heads
        """
        super(ModelNew, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.fused_module = fused_module
        self.eps = self.norm.eps

    def forward(self, x):
        """
        Forward pass of the AttentionBlock.
        :param x: Input tensor of shape (B, C, H, W)
        :return: Output tensor of the same shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Reshape: (B, C, H, W) -> (H*W, B, C)
        x = x.view(B, C, H * W).permute(2, 0, 1).contiguous()
        
        # Self-attention
        attn_output, _ = self.attn(x, x, x)
        
        # Fused residual + layer norm
        x = self.fused_module.fused_residual_layernorm_hip(
            attn_output.contiguous(),
            x.contiguous(),
            self.norm.weight,
            self.norm.bias,
            self.eps
        )
        
        # Reshape back: (H*W, B, C) -> (B, C, H, W)
        x = x.permute(1, 2, 0).view(B, C, H, W)
        return x


def get_inputs():
    return [torch.rand(2, 128, 128, 128).cuda()]


def get_init_inputs():
    return [128, 4]
