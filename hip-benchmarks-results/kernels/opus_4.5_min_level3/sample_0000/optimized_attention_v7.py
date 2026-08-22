import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused residual add + layer norm kernel
fused_kernels_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

// Warp reduce sum using warp shuffle (AMD wavefront is 64 threads)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Single-pass Welford online algorithm for mean and variance
// More numerically stable and only one pass over data
__global__ void fused_residual_layernorm_welford_kernel(
    const float* __restrict__ attn_output,
    const float* __restrict__ residual,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int total_positions,
    int embed_dim,
    float eps
) {
    __shared__ float s_mean[4];
    __shared__ float s_m2[4];
    __shared__ int s_count[4];
    __shared__ float final_mean;
    __shared__ float final_inv_std;
    
    int pos_idx = blockIdx.x;
    if (pos_idx >= total_positions) return;
    
    int base_idx = pos_idx * embed_dim;
    int tid = threadIdx.x;
    int warp_id = tid / 64;
    int lane_id = tid % 64;
    
    // Welford's algorithm - single pass mean and variance
    float mean = 0.0f;
    float m2 = 0.0f;
    int count = 0;
    
    for (int i = tid; i < embed_dim; i += blockDim.x) {
        float val = attn_output[base_idx + i] + residual[base_idx + i];
        count++;
        float delta = val - mean;
        mean += delta / count;
        float delta2 = val - mean;
        m2 += delta * delta2;
    }
    
    // Parallel reduction of Welford stats
    // First reduce within warp
    for (int offset = 32; offset > 0; offset /= 2) {
        float other_mean = __shfl_xor(mean, offset);
        float other_m2 = __shfl_xor(m2, offset);
        int other_count = __shfl_xor(count, offset);
        
        if (count + other_count > 0) {
            int new_count = count + other_count;
            float delta = other_mean - mean;
            mean = (count * mean + other_count * other_mean) / new_count;
            m2 = m2 + other_m2 + delta * delta * count * other_count / new_count;
            count = new_count;
        }
    }
    
    if (lane_id == 0) {
        s_mean[warp_id] = mean;
        s_m2[warp_id] = m2;
        s_count[warp_id] = count;
    }
    __syncthreads();
    
    // Final reduction across warps (only warp 0)
    if (warp_id == 0 && lane_id < 4) {
        mean = s_mean[lane_id];
        m2 = s_m2[lane_id];
        count = s_count[lane_id];
        
        for (int offset = 2; offset > 0; offset /= 2) {
            float other_mean = __shfl_xor(mean, offset);
            float other_m2 = __shfl_xor(m2, offset);
            int other_count = __shfl_xor(count, offset);
            
            if (count + other_count > 0) {
                int new_count = count + other_count;
                float delta = other_mean - mean;
                mean = (count * mean + other_count * other_mean) / new_count;
                m2 = m2 + other_m2 + delta * delta * count * other_count / new_count;
                count = new_count;
            }
        }
        
        if (lane_id == 0) {
            final_mean = mean;
            final_inv_std = rsqrtf(m2 / embed_dim + eps);
        }
    }
    __syncthreads();
    
    float mu = final_mean;
    float inv_std = final_inv_std;
    
    // Apply normalization
    for (int i = tid; i < embed_dim; i += blockDim.x) {
        float val = attn_output[base_idx + i] + residual[base_idx + i];
        float normalized = (val - mu) * inv_std;
        output[base_idx + i] = normalized * gamma[i] + beta[i];
    }
}

torch::Tensor fused_residual_layernorm_hip(
    torch::Tensor attn_output,
    torch::Tensor residual,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
) {
    int total_positions = attn_output.size(0) * attn_output.size(1);
    int embed_dim = attn_output.size(2);
    
    auto output = torch::empty_like(attn_output);
    
    int block_size = 256;
    int num_blocks = total_positions;
    
    fused_residual_layernorm_welford_kernel<<<num_blocks, block_size>>>(
        attn_output.data_ptr<float>(),
        residual.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        total_positions,
        embed_dim,
        eps
    );
    
    return output;
}
"""

fused_kernels_cpp = """
torch::Tensor fused_residual_layernorm_hip(
    torch::Tensor attn_output,
    torch::Tensor residual,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
);
"""

fused_module = load_inline(
    name="fused_attention_kernels_v7",
    cpp_sources=fused_kernels_cpp,
    cuda_sources=fused_kernels_source,
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
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
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
        seq_len = H * W
        
        # Reshape: (B, C, H, W) -> (H*W, B, C)
        x_reshaped = x.view(B, C, seq_len).permute(2, 0, 1).contiguous()
        
        # Use efficient MHA - need_weights=False enables flash attention
        attn_output, _ = self.attn(x_reshaped, x_reshaped, x_reshaped, need_weights=False)
        
        # Fused residual + layer norm
        out = self.fused_module.fused_residual_layernorm_hip(
            attn_output.contiguous(),
            x_reshaped,
            self.norm.weight,
            self.norm.bias,
            self.eps
        )
        
        # Reshape back: (H*W, B, C) -> (B, C, H, W)
        out = out.permute(1, 2, 0).view(B, C, H, W)
        return out


def get_inputs():
    return [torch.rand(2, 128, 128, 128).cuda()]


def get_init_inputs():
    return [128, 4]
