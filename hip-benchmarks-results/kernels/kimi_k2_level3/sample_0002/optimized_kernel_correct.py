import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

os.environ["CXX"] = "hipcc"

# Create a fused MLP + GELU kernel
fused_mlp_gelu_cpp_source = """
#include <hip/hip_runtime.h>

#define GELU_SCALING 0.044715f
#define SQRT_2_OVER_PI 0.7978845608028654f

__global__ void fused_mlp_gelu_kernel(
    const float* x,
    const float* fc1_weight,
    const float* fc1_bias,
    const float* fc2_weight,
    const float* fc2_bias,
    float* out,
    int B, int T, int n_embd, int hidden_size
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int out_dim = threadIdx.x;
    
    if (seq_idx >= T || out_dim >= n_embd) return;
    
    // Calculate fc1 output (x @ fc1_weight^T + fc1_bias)
    int total_threads = blockDim.y * gridDim.y * B;
    float fc1_out = 0.0f;
    for (int i = 0; i < total_threads; ++i) {
        int feature_idx = i % hidden_size;
        int data_idx = (batch_idx * T + seq_idx) * n_embd + feature_idx % n_embd;
        int weight_idx = feature_idx * n_embd + out_dim;
        fc1_out += x[data_idx] * fc1_weight[weight_idx];
    }
    fc1_out += fc1_bias[out_dim];
    
    // Apply GELU
    float x3 = fc1_out * fc1_out * fc1_out;
    float inner = GELU_SCALING * x3 + fc1_out;
    float tanh_val = tanhf(SQRT_2_OVER_PI * inner);
    float gelu_out = 0.5f * fc1_out * (1.0f + tanh_val);
    
    // Store intermediate result (not yet fc2)
    int inter_idx = (batch_idx * T + seq_idx) * n_embd + out_dim;
    out[inter_idx] = gelu_out;
}

torch::Tensor fused_mlp_gelu_hip(
    torch::Tensor x,
    torch::Tensor fc1_weight,
    torch::Tensor fc1_bias,
    torch::Tensor fc2_weight,
    torch::Tensor fc2_bias
) {
    auto B = x.size(0);
    auto T = x.size(1);
    auto n_embd = x.size(2);
    auto hidden_size = fc1_bias.size(0);
    
    auto out = torch::zeros_like(x);
    
    dim3 grid(B, (T + 15) / 16);
    dim3 block(n_embd, 16);
    
    fused_mlp_gelu_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        fc1_weight.data_ptr<float>(),
        fc1_bias.data_ptr<float>(),
        fc2_weight.data_ptr<float>(),
        fc2_bias.data_ptr<float>(),
        out.data_ptr<float>(),
        B, T, n_embd, hidden_size
    );
    
    return out;
}
"""

fused_mlp_gelu = load_inline(
    name="fused_mlp_gelu",
    cpp_sources=fused_mlp_gelu_cpp_source,
    functions=["fused_mlp_gelu_hip"],
    verbose=True,
)

class OptimizedCausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        
        # Use torch.nn.MultiheadAttention for optimized implementation
        self.mha = nn.MultiheadAttention(n_embd, n_head, dropout=attn_pdrop, batch_first=True)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))

    def forward(self, x):
        B, T, C = x.size()
        
        # Use Flash Attention available in PyTorch 2.0+
        y, _ = self.mha(x, x, x, need_weights=False)
        return self.resid_dropout(self.c_proj(y))

class FusedMLP(nn.Module):
    def __init__(self, n_embd, resid_pdrop):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.fused_kernel = fused_mlp_gelu
        self.dropout = nn.Dropout(resid_pdrop)
    
    def forward(self, x):
        # Use fused kernel for fc + gelu, then do fc2
        x_fused = self.fused_kernel.fused_mlp_gelu_hip(
            x, self.c_fc.weight, self.c_fc.bias, self.c_proj.weight, self.c_proj.bias
        )
        # Apply second linear layer separately for correctness
        return self.dropout(self.c_proj(x_fused))

class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = OptimizedCausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = FusedMLP(n_embd, resid_pdrop)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0

def get_inputs():
    return [torch.rand(batch_size, seq_len, n_embd, device='cuda', dtype=torch.float32)]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]