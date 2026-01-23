import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused Residual Add + LayerNorm kernel
fused_residual_layernorm_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Fused residual add + LayerNorm kernel
// Input: x (attention output), residual, weight, bias
// Output: LayerNorm(x + residual)
__global__ void fused_residual_layernorm_kernel(
    const float* __restrict__ attn_out,
    const float* __restrict__ residual,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int seq_len,
    int batch_size,
    int embed_dim,
    float eps
) {
    // Each block handles one (seq, batch) pair
    int idx = blockIdx.x;
    int seq_idx = idx / batch_size;
    int batch_idx = idx % batch_size;
    
    if (seq_idx >= seq_len || batch_idx >= batch_size) return;
    
    int tid = threadIdx.x;
    int base_offset = (seq_idx * batch_size + batch_idx) * embed_dim;
    
    // Shared memory for reduction
    extern __shared__ float shared[];
    float* s_sum = shared;
    float* s_sum_sq = shared + blockDim.x;
    
    // Step 1: Compute sum and sum of squares for mean and variance
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    for (int i = tid; i < embed_dim; i += blockDim.x) {
        float val = attn_out[base_offset + i] + residual[base_offset + i];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }
    
    float mean = s_sum[0] / embed_dim;
    float variance = s_sum_sq[0] / embed_dim - mean * mean;
    float inv_std = rsqrtf(variance + eps);
    
    __syncthreads();
    
    // Step 2: Normalize and apply affine transformation
    for (int i = tid; i < embed_dim; i += blockDim.x) {
        float val = attn_out[base_offset + i] + residual[base_offset + i];
        float normalized = (val - mean) * inv_std;
        out[base_offset + i] = normalized * weight[i] + bias[i];
    }
}

torch::Tensor fused_residual_layernorm_hip(
    torch::Tensor attn_out,
    torch::Tensor residual,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
) {
    // Input shape: (seq_len, batch_size, embed_dim)
    int seq_len = attn_out.size(0);
    int batch_size = attn_out.size(1);
    int embed_dim = attn_out.size(2);
    
    auto out = torch::empty_like(attn_out);
    
    int num_blocks = seq_len * batch_size;
    int block_size = 256;
    int shared_mem_size = 2 * block_size * sizeof(float);
    
    fused_residual_layernorm_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        attn_out.data_ptr<float>(),
        residual.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        seq_len,
        batch_size,
        embed_dim,
        eps
    );
    
    return out;
}
"""

fused_residual_layernorm_cpp = """
torch::Tensor fused_residual_layernorm_hip(
    torch::Tensor attn_out,
    torch::Tensor residual,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
);
"""

fused_module = load_inline(
    name="fused_residual_layernorm",
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
        x = x.view(B, C, H * W).permute(2, 0, 1)  # (seq_len, batch_size, embed_dim)
        
        attn_output, _ = self.attn(x, x, x)
        
        # Fused residual add + LayerNorm
        x = self.fused_module.fused_residual_layernorm_hip(
            attn_output.contiguous(), 
            x.contiguous(),
            self.norm.weight,
            self.norm.bias,
            self.eps
        )
        
        x = x.permute(1, 2, 0).view(B, C, H, W)
        return x
