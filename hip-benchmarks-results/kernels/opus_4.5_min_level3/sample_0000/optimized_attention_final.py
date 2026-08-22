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

// Fused residual add + layer norm kernel - batch_first layout: (B, S, C)
__global__ void fused_residual_layernorm_batch_first_kernel(
    const float* __restrict__ attn_output,
    const float* __restrict__ residual,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int total_positions,
    int embed_dim,
    float eps
) {
    __shared__ float shared_sum[4];
    __shared__ float shared_mean;
    __shared__ float shared_inv_std;
    
    int pos_idx = blockIdx.x;
    if (pos_idx >= total_positions) return;
    
    int base_idx = pos_idx * embed_dim;
    int tid = threadIdx.x;
    int warp_id = tid / 64;
    int lane_id = tid % 64;
    
    // Step 1: Compute sum (for mean)
    float local_sum = 0.0f;
    for (int i = tid; i < embed_dim; i += blockDim.x) {
        local_sum += attn_output[base_idx + i] + residual[base_idx + i];
    }
    
    // Warp reduce
    local_sum = warp_reduce_sum(local_sum);
    
    if (lane_id == 0) {
        shared_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < blockDim.x / 64; i++) {
            total += shared_sum[i];
        }
        shared_mean = total / embed_dim;
    }
    __syncthreads();
    
    float mean = shared_mean;
    
    // Step 2: Compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < embed_dim; i += blockDim.x) {
        float val = attn_output[base_idx + i] + residual[base_idx + i];
        float diff = val - mean;
        var_sum += diff * diff;
    }
    
    var_sum = warp_reduce_sum(var_sum);
    
    if (lane_id == 0) {
        shared_sum[warp_id] = var_sum;
    }
    __syncthreads();
    
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < blockDim.x / 64; i++) {
            total += shared_sum[i];
        }
        shared_inv_std = rsqrtf(total / embed_dim + eps);
    }
    __syncthreads();
    
    float inv_std = shared_inv_std;
    
    // Step 3: Normalize and apply affine transform
    for (int i = tid; i < embed_dim; i += blockDim.x) {
        float val = attn_output[base_idx + i] + residual[base_idx + i];
        float normalized = (val - mean) * inv_std;
        output[base_idx + i] = normalized * gamma[i] + beta[i];
    }
}

torch::Tensor fused_residual_layernorm_batch_first_hip(
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
    
    fused_residual_layernorm_batch_first_kernel<<<num_blocks, block_size>>>(
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
torch::Tensor fused_residual_layernorm_batch_first_hip(
    torch::Tensor attn_output,
    torch::Tensor residual,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
);
"""

fused_module = load_inline(
    name="fused_attention_kernels_final",
    cpp_sources=fused_kernels_cpp,
    cuda_sources=fused_kernels_source,
    functions=["fused_residual_layernorm_batch_first_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Attention Block using Multihead Self-Attention with fused operations.
        Uses batch_first=True layout for better memory access patterns.
        """
        super(ModelNew, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
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
        
        # Reshape: (B, C, H, W) -> (B, H*W, C)
        x_reshaped = x.view(B, C, seq_len).permute(0, 2, 1).contiguous()
        
        # Use efficient MHA with batch_first=True, need_weights=False enables flash attention
        attn_output, _ = self.attn(x_reshaped, x_reshaped, x_reshaped, need_weights=False)
        
        # Fused residual + layer norm on (B, S, C) layout
        out = self.fused_module.fused_residual_layernorm_batch_first_hip(
            attn_output.contiguous(),
            x_reshaped,
            self.norm.weight,
            self.norm.bias,
            self.eps
        )
        
        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out


def get_inputs():
    return [torch.rand(2, 128, 128, 128).cuda()]


def get_init_inputs():
    return [128, 4]
