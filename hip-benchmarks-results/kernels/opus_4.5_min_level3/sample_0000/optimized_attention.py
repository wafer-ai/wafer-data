import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused residual add + layer norm kernel
fused_residual_layernorm_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

// Warp reduce sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 32; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Block reduce sum using shared memory
__device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % 64;
    int wid = threadIdx.x / 64;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 64) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    
    return val;
}

// Fused residual add + layer norm kernel
// Input shape: (seq_len, batch_size, embed_dim)
// Each block handles one (seq_pos, batch) pair
__global__ void fused_residual_layernorm_kernel(
    const float* __restrict__ attn_output,
    const float* __restrict__ residual,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int seq_len,
    int batch_size,
    int embed_dim,
    float eps
) {
    __shared__ float shared_mem[16];
    
    int seq_batch_idx = blockIdx.x;
    int seq_idx = seq_batch_idx / batch_size;
    int batch_idx = seq_batch_idx % batch_size;
    
    if (seq_idx >= seq_len) return;
    
    int base_idx = seq_idx * batch_size * embed_dim + batch_idx * embed_dim;
    
    // Step 1: Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
        float val = attn_output[base_idx + i] + residual[base_idx + i];
        sum += val;
    }
    sum = block_reduce_sum(sum, shared_mem);
    __syncthreads();
    
    __shared__ float mean_shared;
    if (threadIdx.x == 0) {
        mean_shared = sum / embed_dim;
    }
    __syncthreads();
    float mean = mean_shared;
    
    // Step 2: Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
        float val = attn_output[base_idx + i] + residual[base_idx + i];
        float diff = val - mean;
        var_sum += diff * diff;
    }
    var_sum = block_reduce_sum(var_sum, shared_mem);
    __syncthreads();
    
    __shared__ float inv_std_shared;
    if (threadIdx.x == 0) {
        inv_std_shared = rsqrtf(var_sum / embed_dim + eps);
    }
    __syncthreads();
    float inv_std = inv_std_shared;
    
    // Step 3: Normalize and apply scale/bias
    for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
        float val = attn_output[base_idx + i] + residual[base_idx + i];
        float normalized = (val - mean) * inv_std;
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
    auto seq_len = attn_output.size(0);
    auto batch_size = attn_output.size(1);
    auto embed_dim = attn_output.size(2);
    
    auto output = torch::empty_like(attn_output);
    
    int num_blocks = seq_len * batch_size;
    int block_size = 256;
    
    fused_residual_layernorm_kernel<<<num_blocks, block_size>>>(
        attn_output.data_ptr<float>(),
        residual.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
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
