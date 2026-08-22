import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

masked_softmax_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void masked_softmax_kernel(const float* input, float* output, int B, int nh, int T) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    
    // Process each row in parallel
    int row = blockIdx.y % T;
    int head_idx = b * nh + h / T;
    
    int row_offset = (head_idx * T + row) * T;
    
    // Find max and sum simultaneously in one pass
    float max_val = -1e20f;
    float sum_val = 0.0f;
    
    for (int i = threadIdx.x; i < T; i += blockDim.x) {
        if (i <= row) {  // Causal mask: only attend to positions <= current position
            float val = input[row_offset + i];
            if (val > max_val) max_val = val;
        }
    }
    
    // Warp reduction for max
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down(max_val, offset));
        
        float other_sum = __shfl_down(sum_val, offset);
        if (threadIdx.x % 32 == 0) {
            sum_val = 0.0f;  // Only one thread per warp keeps the sum
        }
    }
    
    __shared__ float shared_max;
    if (threadIdx.x == 0) shared_max = max_val;
    __syncthreads();
    float final_max = shared_max;
    
    // Compute exponentials and sum
    float local_exp_sum = 0.0f;
    for (int i = threadIdx.x; i < T; i += blockDim.x) {
        if (i <= row) {
            float val = input[row_offset + i];
            float exp_val = expf(val - final_max);
            output[row_offset + i] = exp_val;
            local_exp_sum += exp_val;
        } else {
            output[row_offset + i] = 0.0f;
        }
    }
    
    // Warp reduction for sum
    for (int offset = 16; offset > 0; offset /= 2) {
        local_exp_sum += __shfl_down(local_exp_sum, offset);
    }
    
    __shared__ float shared_sum;
    if (threadIdx.x % 32 == 0) {
        atomicAdd(&shared_sum, local_exp_sum);
    }
    __syncthreads();
    
    // Normalize
    if (shared_sum > 1e-10f) {
        for (int i = threadIdx.x; i < T; i += blockDim.x) {
            if (i <= row) {
                output[row_offset + i] /= shared_sum;
            }
        }
    }
}

torch::Tensor masked_softmax_hip(torch::Tensor input) {
    int B = input.size(0);
    int nh = input.size(1);
    int T = input.size(2);
    
    auto output = torch::zeros_like(input);
    
    dim3 grid_size(B, nh, 1);
    dim3 block_size(256, 1, 1);
    
    masked_softmax_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, nh, T
    );
    
    return output;
}
"""

masked_softmax = load_inline(
    name="masked_softmax",
    cpp_sources=masked_softmax_cpp_source,
    functions=["masked_softmax_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized multi-head masked self-attention layer with masked softmax kernel.
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
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd
        self.masked_softmax = masked_softmax

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply masked softmax with kernel
        att = self.masked_softmax.masked_softmax_hip(att)
        
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


def get_inputs():
    return [torch.rand(128, 512, 768)]

def get_init_inputs():
    return [768, 8, 0.0, 0.0, 1024]