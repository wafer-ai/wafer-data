import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

# Robust optimized HIP kernel for Vision Attention
vision_attention_fused_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define BLOCK_SIZE 256

// Simple kernel for efficient reshape and transpose
__global__ void prepare_attention_input_kernel(
    const float* input,    // (B, C, H, W)
    float* output,         // (seq_len, B, C)
    int B, int C, int H, int W, int seq_len) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * C * seq_len;
    
    if (idx < total_elements) {
        // Calculate (b, c, seq_pos) from linear index
        int b = idx / (C * seq_len);
        int tmp = idx % (C * seq_len);
        int c = tmp / seq_len;
        int seq_pos = tmp % seq_len;
        
        // Map seq_pos to (h, w)
        int h = seq_pos / W;
        int w = seq_pos % W;
        
        // Compute source index in (B, C, H, W) layout
        int src_idx = ((b * C + c) * H + h) * W + w;
        output[idx] = input[src_idx];
    }
}

// Kernel to fuse residual add and LayerNorm (per token)
__global__ void fused_residual_layernorm_kernel(
    const float* attn_output,  // (seq_len, B, C)
    const float* residual,     // (seq_len, B, C)
    const float* weight,       // (C,)
    const float* bias,         // (C,)
    float* output,             // (seq_len, B, C)
    int seq_len, int B, int C) {
    
    // Each block processes one token (one position in the sequence)
    int token_idx = blockIdx.x;
    int total_tokens = seq_len * B;
    
    if (token_idx >= total_tokens) return;
    
    // Each thread processes one channel element
    int c_idx = threadIdx.x;
    
    // Offset for this token
    int offset = token_idx * C;
    
    // Compute residual addition
    if (c_idx < C) {
        float val = attn_output[offset + c_idx] + residual[offset + c_idx];
        
        // Compute mean and var using cooperative approach
        // Simplified: use atomic operations for reduction (less efficient but correct)
        __shared__ float shared_mean;
        __shared__ float shared_inv_std;
        __shared__ float shared_sum;
        __shared__ float shared_var_sum;
        
        if (threadIdx.x == 0) {
            shared_sum = 0.0f;
            shared_var_sum = 0.0f;
        }
        __syncthreads();
        
        // Accumulate sum for mean
        atomicAdd(&shared_sum, val);
        __syncthreads();
        
        // Calculate mean
        if (threadIdx.x == 0) {
            shared_mean = shared_sum / C;
        }
        __syncthreads();
        
        // Accumulate variance
        float diff = val - shared_mean;
        atomicAdd(&shared_var_sum, diff * diff);
        __syncthreads();
        
        // Calculate inverse std
        if (threadIdx.x == 0) {
            float variance = shared_var_sum / C;
            shared_inv_std = rsqrtf(variance + 1e-5f);
        }
        __syncthreads();
        
        // Apply LayerNorm
        float normalized = (val - shared_mean) * shared_inv_std;
        output[offset + c_idx] = normalized * weight[c_idx] + bias[c_idx];
    }
}

// Separate kernel for final reshape and transpose
__global__ void final_reshape_kernel(
    const float* input,    // (seq_len, B, C)
    float* output,         // (B, C, H, W)
    int seq_len, int B, int C, int H, int W) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * C * seq_len;
    
    if (idx < total_elements) {
        // Calculate (b, c, seq_pos) from linear index
        int b = idx / (C * seq_len);
        int tmp = idx % (C * seq_len);
        int c = tmp / seq_len;
        int seq_pos = tmp % seq_len;
        
        // Map seq_pos to (h, w)
        int h = seq_pos / W;
        int w = seq_pos % W;
        
        // Compute destination index in (B, C, H, W) layout
        int dst_idx = ((b * C + c) * H + h) * W + w;
        output[dst_idx] = input[idx];
    }
}

torch::Tensor prepare_attention_input(torch::Tensor input) {
    int B = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int seq_len = H * W;
    
    auto output = torch::zeros({seq_len, B, C}, input.options());
    
    int total_elements = B * C * seq_len;
    int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    prepare_attention_input_kernel<<<num_blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W, seq_len);
    
    return output;
}

torch::Tensor fused_residual_layernorm(
    torch::Tensor attn_output,
    torch::Tensor residual,
    torch::Tensor weight,
    torch::Tensor bias) {
    
    int seq_len = attn_output.size(0);
    int B = attn_output.size(1);
    int C = attn_output.size(2);
    
    auto output = torch::zeros_like(attn_output);
    
    int num_blocks = seq_len * B;
    int threads_per_block = C;
    
    fused_residual_layernorm_kernel<<<num_blocks, threads_per_block>>>(
        attn_output.data_ptr<float>(),
        residual.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        seq_len, B, C);
    
    return output;
}

torch::Tensor final_reshape(torch::Tensor input, int H, int W) {
    int seq_len = input.size(0);
    int B = input.size(1);
    int C = input.size(2);
    
    auto output = torch::zeros({B, C, H, W}, input.options());
    
    int total_elements = B * C * seq_len;
    int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    final_reshape_kernel<<<num_blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        seq_len, B, C, H, W);
    
    return output;
}
"""

# Compile the optimized kernels
vision_kernels = load_inline(
    name="vision_kernels",
    cpp_sources=vision_attention_fused_source,
    functions=["prepare_attention_input", "fused_residual_layernorm", "final_reshape"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Core attention (highly optimized in PyTorch)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        
        # LayerNorm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Custom kernels
        self.kernels = vision_kernels
        
    def forward(self, x):
        B, C, H, W = x.shape
        seq_len = H * W
        
        # Optimized reshape: (B, C, H, W) -> (seq_len, B, C)
        x = self.kernels.prepare_attention_input(x)
        residual = x.clone()
        
        # Apply multi-head attention
        attn_output, _ = self.attn(x, x, x)
        
        # Fused residual add + LayerNorm
        x = self.kernels.fused_residual_layernorm(
            attn_output, residual, self.norm.weight, self.norm.bias
        )
        
        # Optimized reshape: (seq_len, B, C) -> (B, C, H, W)
        x = self.kernels.final_reshape(x, H, W)
        
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