
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Define the C++ kernel source
cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define WARP_SIZE 64

// --- Helper Functions for Reduction ---
__device__ float warpReduceMax(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val = max(val, __shfl_down(val, offset));
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down(val, offset);
    return val;
}

// --- Fused Causal Softmax Kernel ---
// Fuses: Scale -> Mask (Causal) -> Softmax
__global__ void fused_causal_softmax_kernel(float* __restrict__ att, int T, float scale) {
    // att: (N, T, T) flat, where N = B * nh
    // Grid: (N * T, 1, 1). Block: (1024, 1, 1).
    // Each block handles one row of length T.
    
    int row_idx_global = blockIdx.x;
    int row_in_seq = row_idx_global % T;
    
    // Pointer to the start of the row
    float* row_ptr = att + row_idx_global * T;
    
    int tid = threadIdx.x;
    
    // 1. Load and Mask
    float val = -INFINITY;
    if (tid < T) {
        val = row_ptr[tid];
        val *= scale; // Scale
        if (tid > row_in_seq) { // Causal Mask
            val = -INFINITY;
        }
    }
    
    // 2. Reduce Max (for numerical stability)
    static __shared__ float shared_max[32]; // Max 32 warps (1024/32 = 32)
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    
    float warp_max = warpReduceMax(val);
    if (lane == 0) shared_max[wid] = warp_max;
    __syncthreads();
    
    float block_max = -INFINITY;
    if (tid < num_warps) block_max = shared_max[tid];
    if (wid == 0) {
        if (tid >= num_warps) block_max = -INFINITY;
        block_max = warpReduceMax(block_max);
    }
    if (tid == 0) shared_max[0] = block_max;
    __syncthreads();
    block_max = shared_max[0];
    
    // 3. Compute Exp
    float exp_val = 0.0f;
    if (tid < T) {
        // If val is -inf, exp is 0.
        exp_val = expf(val - block_max);
    }
    
    // 4. Reduce Sum
    static __shared__ float shared_sum[32];
    float warp_sum = warpReduceSum(exp_val);
    if (lane == 0) shared_sum[wid] = warp_sum;
    __syncthreads();
    
    float block_sum = 0.0f;
    if (tid < num_warps) block_sum = shared_sum[tid];
    if (wid == 0) {
        if (tid >= num_warps) block_sum = 0.0f;
        block_sum = warpReduceSum(block_sum);
    }
    if (tid == 0) shared_sum[0] = block_sum;
    __syncthreads();
    block_sum = shared_sum[0];
    
    // 5. Write Output
    if (tid < T) {
        // Avoid division by zero if all -inf (should not happen in causal due to diagonal)
        // But for safety
        if (block_sum > 1e-6f)
            row_ptr[tid] = exp_val / block_sum;
        else
            row_ptr[tid] = 0.0f;
    }
}

// --- New GELU Kernel ---
__global__ void new_gelu_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        const float k_sqrt_2_over_pi = 0.7978845608028654f;
        const float k_coeff = 0.044715f;
        
        float x_cubed = x * x * x;
        float inner = k_sqrt_2_over_pi * (x + k_coeff * x_cubed);
        float tanh_val = tanhf(inner);
        
        output[idx] = 0.5f * x * (1.0f + tanh_val);
    }
}

// --- Bindings ---

torch::Tensor new_gelu_hip(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    new_gelu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    return output;
}

torch::Tensor causal_softmax_hip(torch::Tensor att, float scale) {
    // att: (B, nh, T, T)
    int B = att.size(0);
    int nh = att.size(1);
    int T = att.size(2);
    int num_rows = B * nh * T;
    
    // We modify att in-place
    // Launch one block per row
    fused_causal_softmax_kernel<<<num_rows, 1024>>>(
        att.data_ptr<float>(),
        T,
        scale
    );
    return att;
}
"""

# Compile the inline C++ code
module = load_inline(
    name="mini_gpt_ops_v2",
    cpp_sources=cpp_source,
    functions=["new_gelu_hip", "causal_softmax_hip"],
    verbose=True,
    extra_cflags=['-O3']
)

class NewGELUOptimized(nn.Module):
    """
    Optimized implementation of NewGELU using HIP kernel.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return module.new_gelu_hip(x)

class CausalSelfAttentionOptimized(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    Optimized with fused Scaled Masked Softmax kernel.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        # bias buffer not needed for fused kernel, but kept for interface consistency if needed (unused)
        # actually, removing it saves memory.
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        
        # FUSED OPERATION STARTS
        # Original:
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        
        # New:
        att = (q @ k.transpose(-2, -1))
        # Ensure contiguous (matmul usually returns contiguous, but to be safe)
        if not att.is_contiguous():
            att = att.contiguous()
            
        scale = 1.0 / math.sqrt(k.size(-1))
        att = module.causal_softmax_hip(att, scale)
        # FUSED OPERATION ENDS
        
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class ModelNew(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttentionOptimized(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = NewGELUOptimized(), # Replaced here
            dropout = nn.Dropout(resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0

def get_inputs():
    return [torch.rand(batch_size, seq_len, n_embd).cuda()]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]
