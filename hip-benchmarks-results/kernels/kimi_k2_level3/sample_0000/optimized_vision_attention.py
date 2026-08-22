import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

# Optimized HIP kernel that fuses residual addition and LayerNorm
vision_attention_fused_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define BLOCK_SIZE 256

// Kernel to fuse residual add and LayerNorm for each token
__global__ void fused_residual_layernorm_kernel(
    const float* attn_output,  // Input from attention: (seq_len, batch_size, embed_dim)
    const float* residual,     // Residual connection: (seq_len, batch_size, embed_dim)
    const float* weight,       // LayerNorm weight: (embed_dim,)
    const float* bias,         // LayerNorm bias: (embed_dim,)
    float* output,             // Output: (seq_len, batch_size, embed_dim)
    int seq_len, int batch_size, int embed_dim) {
    
    // Each thread block processes one token
    int token_idx = blockIdx.x;
    int total_tokens = seq_len * batch_size;
    
    if (token_idx < total_tokens) {
        int seq_idx = token_idx / batch_size;
        int batch_idx = token_idx % batch_size;
        
        // Calculate offset for this token
        int offset = seq_idx * batch_size * embed_dim + batch_idx * embed_dim;
        
        const float* attn_ptr = attn_output + offset;
        const float* residual_ptr = residual + offset;
        float* out_ptr = output + offset;
        
        // Shared memory for this token's data
        __shared__ float shared_mem[256];  // Assuming embed_dim <= 256
        
        // Load and compute residual addition
        float sum = 0.0f;
        for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
            shared_mem[i] = attn_ptr[i] + residual_ptr[i];
            sum += shared_mem[i];
        }
        
        // Compute mean (simple parallel reduction)
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride && threadIdx.x + stride < embed_dim) {
                sum += shared_mem[threadIdx.x + stride];
            }
            __syncthreads();
        }
        
        if (threadIdx.x == 0) {
            shared_mem[0] = sum / embed_dim;  // Store mean
        }
        __syncthreads();
        
        float mean = shared_mem[0];
        
        // Compute variance
        float var = 0.0f;
        for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
            float diff = shared_mem[i] - mean;
            var += diff * diff;
        }
        
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride && threadIdx.x + stride < embed_dim) {
                var += shared_mem[threadIdx.x + stride];
            }
            __syncthreads();
        }
        
        if (threadIdx.x == 0) {
            shared_mem[0] = var / embed_dim;  // Store variance
        }
        __syncthreads();
        
        float variance = shared_mem[0];
        
        // Apply LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
        float inv_std = rsqrtf(variance + 1e-5f);
        for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
            float normalized = (shared_mem[i] - mean) * inv_std;
            out_ptr[i] = normalized * weight[i] + bias[i];
        }
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
    fused_residual_layernorm_kernel<<<num_blocks, BLOCK_SIZE>>>(
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
        
        # LayerNorm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Fused kernel for residual + norm
        self.vision_attn = vision_attn
        
    def forward(self, x):
        B, C, H, W = x.shape
        seq_len = H * W
        
        # Store residual for connection
        residual = x
        
        # Reshape: (B, C, H, W) -> (seq_len, batch_size, embed_dim)
        x = x.view(B, C, seq_len).permute(2, 0, 1)
        
        # Apply multi-head attention
        attn_output, _ = self.attn(x, x, x)
        
        # Fused residual add + LayerNorm (keeps seq_len dimension)
        x = self.vision_attn.vision_attention_forward(
            attn_output, x, self.norm.weight, self.norm.bias
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