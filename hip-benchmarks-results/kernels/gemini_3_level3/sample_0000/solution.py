import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Set compiler to hipcc
os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>

__global__ void fused_add_layernorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ attn,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int N,
    int D,
    float eps)
{
    extern __shared__ char s_buffer[];
    float* s_reduce = (float*)s_buffer;
    float* s_vals = (float*)&s_reduce[blockDim.x]; 

    int tid = threadIdx.x;
    int row_idx = blockIdx.x;

    if (row_idx >= N) return;

    // Pointers to this row
    const float* row_x = x + row_idx * D;
    const float* row_attn = attn + row_idx * D;
    float* row_out = out + row_idx * D;

    // 1. Load and Compute Mean
    float thread_sum = 0.0f;
    
    // Iterate if D > blockDim.x (unlikely here but safe)
    for (int i = tid; i < D; i += blockDim.x) {
        float val = row_x[i] + row_attn[i];
        s_vals[i] = val; // Store in shared memory for reuse
        thread_sum += val;
    }
    
    s_reduce[tid] = thread_sum;
    __syncthreads();
    
    // Reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_reduce[tid] += s_reduce[tid + s];
        }
        __syncthreads();
    }
    float mean = s_reduce[0] / D;
    __syncthreads(); 

    // 2. Calculate Variance
    float thread_sq_diff = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float val = s_vals[i];
        float diff = val - mean;
        thread_sq_diff += diff * diff;
    }

    s_reduce[tid] = thread_sq_diff;
    __syncthreads();

    // Reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_reduce[tid] += s_reduce[tid + s];
        }
        __syncthreads();
    }
    float var = s_reduce[0] / D;
    float inv_std = rsqrtf(var + eps);

    // 3. Normalize and Write
    for (int i = tid; i < D; i += blockDim.x) {
        float val = s_vals[i];
        float n_val = (val - mean) * inv_std;
        row_out[i] = n_val * gamma[i] + beta[i];
    }
}

torch::Tensor fused_add_layernorm_hip(torch::Tensor x, torch::Tensor attn, torch::Tensor gamma, torch::Tensor beta, float eps) {
    // x, attn are (B, L, D) or (L, B, D). We treat as (N, D).
    int N = x.numel() / x.size(-1);
    int D = x.size(-1);
    
    auto out = torch::empty_like(x);
    
    int block_size = 256;
    int shared_mem_size = (block_size + D) * sizeof(float);
    
    fused_add_layernorm_kernel<<<N, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        attn.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        out.data_ptr<float>(),
        N, D, eps
    );
    
    return out;
}
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=cpp_source,
    functions=["fused_add_layernorm_hip"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(ModelNew, self).__init__()
        # Use batch_first=True for potential optimization
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.fused_ops = fused_ops

    def forward(self, x):
        B, C, H, W = x.shape
        # x: (B, C, H, W)
        # Convert to (B, L, C)
        x = x.flatten(2).transpose(1, 2).contiguous()
        
        # MHA with need_weights=False to disable attention matrix materialization
        attn_output, _ = self.attn(x, x, x, need_weights=False)
        
        # Fused Add + LayerNorm
        # x and attn_output are (B, L, C)
        x = self.fused_ops.fused_add_layernorm_hip(
            x, attn_output, self.norm.weight, self.norm.bias, self.norm.eps
        )
        
        # Convert back to (B, C, H, W)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x
