
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

fused_kernels_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void fused_bias_add_kernel(float* x, const float* bias, int B, int T, int C) {
    int b = blockIdx.z;
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (t < T && c < C) {
        x[b * T * C + t * C + c] += bias[c];
    }
}

torch::Tensor fused_bias_add_hip(torch::Tensor x, torch::Tensor bias) {
    int B = x.size(0);
    int T = x.size(1);
    int C = x.size(2);
    
    dim3 block(32, 32);
    dim3 grid((C + 31) / 32, (T + 31) / 32, B);
    
    fused_bias_add_kernel<<<grid, block>>>(x.data_ptr<float>(), bias.data_ptr<float>(), B, T, C);
    return x;
}
"""

fused_lib = load_inline(
    name="fused_lib",
    cpp_sources=fused_kernels_source,
    functions=["fused_bias_add_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()

        # 1. qkv projection
        qkv = self.c_attn(x)
        
        # 2. Reshape and transpose q, k, v
        # Using more efficient PyTorch view/permute
        q, k, v = qkv.view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4).unbind(0)

        # 3. Optimized scaled dot product attention
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.attn_dropout.p if self.training else 0.0, 
            is_causal=True
        )

        # 4. Transpose and reshape back
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 5. Output projection and fused bias add
        # We manually perform the linear transformation without bias, then add bias with our kernel.
        bias = self.c_proj.bias
        y = F.linear(y, self.c_proj.weight, None)
        y = fused_lib.fused_bias_add_hip(y, bias)
        
        y = self.resid_dropout(y)
        return y
