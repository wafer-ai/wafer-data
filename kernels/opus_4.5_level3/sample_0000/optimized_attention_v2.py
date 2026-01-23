import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused kernels for attention block
fused_kernels_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define WARP_SIZE 64
#define BLOCK_SIZE 256

// Fused residual add + LayerNorm kernel with better memory access
__global__ void fused_residual_layernorm_kernel_v2(
    const float* __restrict__ attn_out,
    const float* __restrict__ residual,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int total_elements,
    int embed_dim,
    float eps
) {
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    
    int base_offset = idx * embed_dim;
    
    // Use warp-level reductions for better performance
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    // Vector load if possible - process 4 elements at a time
    int i = tid;
    while (i < embed_dim) {
        float val = attn_out[base_offset + i] + residual[base_offset + i];
        local_sum += val;
        local_sum_sq += val * val;
        i += blockDim.x;
    }
    
    // Warp-level reduction using shuffle operations
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down(local_sum, offset, WARP_SIZE);
        local_sum_sq += __shfl_down(local_sum_sq, offset, WARP_SIZE);
    }
    
    // Shared memory for cross-warp reduction
    __shared__ float s_sum[BLOCK_SIZE / WARP_SIZE];
    __shared__ float s_sum_sq[BLOCK_SIZE / WARP_SIZE];
    
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    
    if (lane == 0) {
        s_sum[warp_id] = local_sum;
        s_sum_sq[warp_id] = local_sum_sq;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (warp_id == 0) {
        local_sum = (lane < (BLOCK_SIZE / WARP_SIZE)) ? s_sum[lane] : 0.0f;
        local_sum_sq = (lane < (BLOCK_SIZE / WARP_SIZE)) ? s_sum_sq[lane] : 0.0f;
        
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            local_sum += __shfl_down(local_sum, offset, WARP_SIZE);
            local_sum_sq += __shfl_down(local_sum_sq, offset, WARP_SIZE);
        }
    }
    
    __shared__ float mean, inv_std;
    if (tid == 0) {
        mean = local_sum / embed_dim;
        float variance = local_sum_sq / embed_dim - mean * mean;
        inv_std = rsqrtf(variance + eps);
    }
    __syncthreads();
    
    // Apply normalization
    i = tid;
    while (i < embed_dim) {
        float val = attn_out[base_offset + i] + residual[base_offset + i];
        float normalized = (val - mean) * inv_std;
        out[base_offset + i] = normalized * weight[i] + bias[i];
        i += blockDim.x;
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
    int block_size = BLOCK_SIZE;
    
    fused_residual_layernorm_kernel_v2<<<num_blocks, block_size>>>(
        attn_out.data_ptr<float>(),
        residual.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        num_blocks,
        embed_dim,
        eps
    );
    
    return out;
}

// Fused reshape view from (B, C, H, W) to (H*W, B, C) and back
// This combines view + permute into efficient memory copy
__global__ void fused_reshape_to_seq_kernel(
    const float* __restrict__ input,  // (B, C, H, W)
    float* __restrict__ output,       // (H*W, B, C)
    int B, int C, int H, int W
) {
    int seq_len = H * W;
    int total = seq_len * B * C;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    
    // Output index: (seq, batch, channel)
    int channel = idx % C;
    int temp = idx / C;
    int batch = temp % B;
    int seq = temp / B;
    
    // Input index: (batch, channel, h, w) where seq = h * W + w
    int h = seq / W;
    int w = seq % W;
    
    int in_idx = batch * C * H * W + channel * H * W + h * W + w;
    output[idx] = input[in_idx];
}

__global__ void fused_reshape_from_seq_kernel(
    const float* __restrict__ input,  // (H*W, B, C)
    float* __restrict__ output,       // (B, C, H, W)
    int B, int C, int H, int W
) {
    int seq_len = H * W;
    int total = seq_len * B * C;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    
    // Input index: (seq, batch, channel)
    int channel = idx % C;
    int temp = idx / C;
    int batch = temp % B;
    int seq = temp / B;
    
    int h = seq / W;
    int w = seq % W;
    
    // Output index: (batch, channel, h, w)
    int out_idx = batch * C * H * W + channel * H * W + h * W + w;
    output[out_idx] = input[idx];
}

torch::Tensor fused_reshape_to_seq_hip(torch::Tensor input, int H, int W) {
    int B = input.size(0);
    int C = input.size(1);
    int seq_len = H * W;
    
    auto output = torch::empty({seq_len, B, C}, input.options());
    
    int total = seq_len * B * C;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    
    fused_reshape_to_seq_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W
    );
    
    return output;
}

torch::Tensor fused_reshape_from_seq_hip(torch::Tensor input, int B, int C, int H, int W) {
    auto output = torch::empty({B, C, H, W}, input.options());
    
    int total = H * W * B * C;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    
    fused_reshape_from_seq_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W
    );
    
    return output;
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
torch::Tensor fused_reshape_to_seq_hip(torch::Tensor input, int H, int W);
torch::Tensor fused_reshape_from_seq_hip(torch::Tensor input, int B, int C, int H, int W);
"""

fused_module = load_inline(
    name="fused_attention_kernels",
    cpp_sources=fused_kernels_cpp,
    cuda_sources=fused_kernels_source,
    functions=["fused_residual_layernorm_hip", "fused_reshape_to_seq_hip", "fused_reshape_from_seq_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"],
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
        
        # Fused reshape: (B, C, H, W) -> (H*W, B, C)
        x_seq = self.fused_module.fused_reshape_to_seq_hip(x.contiguous(), H, W)
        
        attn_output, _ = self.attn(x_seq, x_seq, x_seq)
        
        # Fused residual add + LayerNorm
        x_norm = self.fused_module.fused_residual_layernorm_hip(
            attn_output.contiguous(), 
            x_seq.contiguous(),
            self.norm.weight,
            self.norm.bias,
            self.eps
        )
        
        # Fused reshape: (H*W, B, C) -> (B, C, H, W)
        out = self.fused_module.fused_reshape_from_seq_hip(x_norm, B, C, H, W)
        
        return out
