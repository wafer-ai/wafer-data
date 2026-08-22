import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused residual add + layer norm kernel (optimized)
fused_residual_layernorm_source = """
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

// Fused residual add + layer norm kernel
// Processes one sequence position per block
__global__ void fused_residual_layernorm_kernel(
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
    
    fused_residual_layernorm_kernel<<<num_blocks, block_size>>>(
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
    name="fused_residual_layernorm_v3",
    cpp_sources=fused_residual_layernorm_cpp,
    cuda_sources=fused_residual_layernorm_source,
    functions=["fused_residual_layernorm_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Attention Block using efficient scaled_dot_product_attention with fused operations.
        """
        super(ModelNew, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Separate Q, K, V projections for better control
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
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
        
        # Reshape: (B, C, H, W) -> (B, seq_len, C)
        x_reshaped = x.view(B, C, seq_len).permute(0, 2, 1).contiguous()
        
        # Project Q, K, V
        q = self.q_proj(x_reshaped)  # (B, seq_len, embed_dim)
        k = self.k_proj(x_reshaped)
        v = self.v_proj(x_reshaped)
        
        # Reshape for multi-head attention: (B, seq_len, num_heads, head_dim) -> (B, num_heads, seq_len, head_dim)
        q = q.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Efficient attention (uses Flash Attention when available)
        attn_output = F.scaled_dot_product_attention(q, k, v)
        
        # Reshape back: (B, num_heads, seq_len, head_dim) -> (B, seq_len, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, seq_len, self.embed_dim)
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        
        # Convert to (seq_len, B, embed_dim) for layer norm
        attn_output = attn_output.permute(1, 0, 2).contiguous()
        x_residual = x_reshaped.permute(1, 0, 2).contiguous()
        
        # Fused residual + layer norm
        out = self.fused_module.fused_residual_layernorm_hip(
            attn_output,
            x_residual,
            self.norm.weight,
            self.norm.bias,
            self.eps
        )
        
        # Reshape back: (seq_len, B, C) -> (B, C, H, W)
        out = out.permute(1, 2, 0).view(B, C, H, W)
        return out


def get_inputs():
    return [torch.rand(2, 128, 128, 128).cuda()]


def get_init_inputs():
    return [128, 4]
