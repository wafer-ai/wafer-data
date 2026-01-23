import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernels for the attention block
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

// Fused residual add + layer norm kernel
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

// Fused reshape and permute: (B, C, H, W) -> (H*W, B, C) 
// Each thread handles one element
__global__ void reshape_bchw_to_sbc_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    if (idx >= total) return;
    
    // Decode index for output: (s, b, c) where s = h*W + w
    int S = H * W;
    int c = idx % C;
    int b = (idx / C) % B;
    int s = idx / (C * B);
    
    if (s >= S) return;
    
    int h = s / W;
    int w = s % W;
    
    // Input index: (b, c, h, w)
    int in_idx = b * (C * H * W) + c * (H * W) + h * W + w;
    
    output[idx] = input[in_idx];
}

// Fused reshape and permute: (S, B, C) -> (B, C, H, W)
__global__ void reshape_sbc_to_bchw_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    if (idx >= total) return;
    
    // Decode index for output: (b, c, h, w)
    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int b = idx / (W * H * C);
    
    // Input index: (s, b, c) where s = h*W + w
    int s = h * W + w;
    int S = H * W;
    int in_idx = s * (B * C) + b * C + c;
    
    output[idx] = input[in_idx];
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

torch::Tensor reshape_bchw_to_sbc_hip(torch::Tensor input) {
    int B = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int S = H * W;
    
    auto output = torch::empty({S, B, C}, input.options());
    
    int total = S * B * C;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    
    reshape_bchw_to_sbc_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W
    );
    
    return output;
}

torch::Tensor reshape_sbc_to_bchw_hip(torch::Tensor input, int H, int W) {
    int S = input.size(0);
    int B = input.size(1);
    int C = input.size(2);
    
    auto output = torch::empty({B, C, H, W}, input.options());
    
    int total = B * C * H * W;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    
    reshape_sbc_to_bchw_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W
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
torch::Tensor reshape_bchw_to_sbc_hip(torch::Tensor input);
torch::Tensor reshape_sbc_to_bchw_hip(torch::Tensor input, int H, int W);
"""

fused_module = load_inline(
    name="fused_attention_kernels_v5",
    cpp_sources=fused_kernels_cpp,
    cuda_sources=fused_kernels_source,
    functions=["fused_residual_layernorm_hip", "reshape_bchw_to_sbc_hip", "reshape_sbc_to_bchw_hip"],
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
        
        # Reshape: (B, C, H, W) -> (H*W, B, C) using custom kernel
        x_reshaped = self.fused_module.reshape_bchw_to_sbc_hip(x.contiguous())
        
        # Self-attention using PyTorch's efficient MHA
        attn_output, _ = self.attn(x_reshaped, x_reshaped, x_reshaped, need_weights=False)
        
        # Fused residual + layer norm
        out = self.fused_module.fused_residual_layernorm_hip(
            attn_output.contiguous(),
            x_reshaped,
            self.norm.weight,
            self.norm.bias,
            self.eps
        )
        
        # Reshape back: (H*W, B, C) -> (B, C, H, W) using custom kernel
        out = self.fused_module.reshape_sbc_to_bchw_hip(out, H, W)
        return out


def get_inputs():
    return [torch.rand(2, 128, 128, 128).cuda()]


def get_init_inputs():
    return [128, 4]
