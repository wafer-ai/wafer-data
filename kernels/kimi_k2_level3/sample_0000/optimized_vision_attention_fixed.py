import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

# Fixed optimized HIP kernel that fuses residual addition and LayerNorm
vision_attention_fused_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Device function for warp reduction
__device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Kernel to fuse residual add and LayerNorm for each token
__global__ void fused_residual_layernorm_kernel(
    const float* attn_output,  // Input from attention: (seq_len, batch_size, embed_dim)
    const float* residual,     // Residual connection: (seq_len, batch_size, embed_dim)
    const float* weight,       // LayerNorm weight: (embed_dim,)
    const float* bias,         // LayerNorm bias: (embed_dim,)
    float* output,             // Output: (seq_len, batch_size, embed_dim)
    int seq_len, int batch_size, int embed_dim) {
    
    // Each block processes one token
    int token_idx = blockIdx.x;
    int total_tokens = seq_len * batch_size;
    
    if (token_idx >= total_tokens) return;
    
    // Calculate offset for this token
    int offset = token_idx * embed_dim;
    
    const float* attn_ptr = attn_output + offset;
    const float* residual_ptr = residual + offset;
    float* out_ptr = output + offset;
    
    // Shared memory for reduction
    __shared__ float shared_sum[BLOCK_SIZE / WARP_SIZE];
    __shared__ float shared_var[BLOCK_SIZE / WARP_SIZE];
    __shared__ float mean;
    __shared__ float inv_std;
    
    // Step 1: Compute residual addition and sum for mean
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
        float val = attn_ptr[i] + residual_ptr[i];
        out_ptr[i] = val;
        local_sum += val;
    }
    
    // Step 2: Compute mean using warp reduction
    local_sum = warp_reduce_sum(local_sum);
    
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    
    if (lane == 0) {
        shared_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (threadIdx.x == 0) {
        float total_sum = 0.0f;
        int num_warps = blockDim.x / WARP_SIZE;
        for (int i = 0; i < num_warps; i++) {
            total_sum += shared_sum[i];
        }
        mean = total_sum / embed_dim;
    }
    __syncthreads();
    
    // Step 3: Compute variance
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
        float diff = out_ptr[i] - mean;
        local_var += diff * diff;
    }
    
    // Warp reduction for variance
    local_var = warp_reduce_sum(local_var);
    
    if (lane == 0) {
        shared_var[warp_id] = local_var;
    }
    __syncthreads();
    
    // Final reduction across warps for variance
    if (threadIdx.x == 0) {
        float total_var = 0.0f;
        int num_warps = blockDim.x / WARP_SIZE;
        for (int i = 0; i < num_warps; i++) {
            total_var += shared_var[i];
        }
        inv_std = rsqrtf(total_var / embed_dim + 1e-5f);
    }
    __syncthreads();
    
    // Step 4: Apply LayerNorm
    for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
        float normalized = (out_ptr[i] - mean) * inv_std;
        out_ptr[i] = normalized * weight[i] + bias[i];
    }
}

torch::Tensor vision_attention_forward(
    torch::Tensor attn_output, torch::Tensor residual,
    torch::Tensor weight, torch::Tensor bias) {
    
    int seq_len = attn_output.size(0);
    int batch_size = attn_output.size(1);
    int embed_dim = attn_output.size(2);
    
    auto output = torch::zeros_like(attn_output);
    
    int num_blocks = seq_len * batch_size;
    int grid_size = num_blocks;
    int block_size = BLOCK_SIZE;
    
    fused_residual_layernorm_kernel<<<grid_size, block_size>>>(
        attn_output.data_ptr<float>(),
        residual.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        seq_len, batch_size, embed_dim);
    
    return output;
}
"""

# Compile the optimized kernel
vision_attn = load_inline(
    name="vision_attn",
    cpp_sources=vision_attention_fused_source,
    functions=["vision_attention_forward"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Core attention (PyTorch's optimized implementation)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        
        # LayerNorm parameters
        self.norm = nn.LayerNorm(embed_dim)
        
        # Fused kernel for residual + norm
        self.vision_attn = vision_attn
        
    def forward(self, x):
        B, C, H, W = x.shape
        seq_len = H * W
        
        # Store residual for connection
        residual = x.view(B, C, seq_len).permute(2, 0, 1)  # (seq_len, batch_size, embed_dim)
        
        # Reshape: (B, C, H, W) -> (seq_len, batch_size, embed_dim)
        x = residual
        
        # Apply multi-head attention
        attn_output, _ = self.attn(x, x, x)
        
        # Fused residual add + LayerNorm
        x = self.vision_attn.vision_attention_forward(
            attn_output, residual, self.norm.weight, self.norm.bias
        )
        
        # Reshape back: (seq_len, batch_size, embed_dim) -> (B, C, H, W)
        x = x.permute(1, 2, 0).view(B, C, H, W)
        
        return x

def get_inputs():
    batch_size = 2
    num_channels = 128
    image_height = 128
    image_width = 128
    return [torch.rand(batch_size, num_channels, image_height, image_width)]

def get_init_inputs():
    embed_dim = 128
    num_heads = 4
    return [embed_dim, num_heads]