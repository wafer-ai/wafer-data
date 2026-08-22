import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# C++ declarations for binding
cpp_source = """
#include <torch/extension.h>

torch::Tensor fused_gelu_hip(torch::Tensor x);
torch::Tensor fused_layernorm_hip(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float eps);
torch::Tensor fused_scale_mask_softmax_hip(torch::Tensor att, torch::Tensor mask, float scale);
"""

# Combined HIP source with optimized kernels
hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>
#include <float.h>

// Vectorized GELU kernel using float4
__global__ void fused_gelu_kernel_vec4(const float4* __restrict__ x, float4* __restrict__ out, int size4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size4) {
        float4 val = x[idx];
        const float sqrt_2_over_pi = 0.7978845608028654f;
        
        float4 result;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float v = ((float*)&val)[i];
            float v3 = v * v * v;
            float inner = sqrt_2_over_pi * (v + 0.044715f * v3);
            ((float*)&result)[i] = 0.5f * v * (1.0f + tanhf(inner));
        }
        out[idx] = result;
    }
}

__global__ void fused_gelu_kernel(const float* __restrict__ x, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        const float sqrt_2_over_pi = 0.7978845608028654f;
        float val_cubed = val * val * val;
        float inner = sqrt_2_over_pi * (val + 0.044715f * val_cubed);
        out[idx] = 0.5f * val * (1.0f + tanhf(inner));
    }
}

torch::Tensor fused_gelu_hip(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    
    // Use vectorized version when possible
    if (size % 4 == 0) {
        const int block_size = 256;
        const int size4 = size / 4;
        const int num_blocks = (size4 + block_size - 1) / block_size;
        
        fused_gelu_kernel_vec4<<<num_blocks, block_size>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()), 
            reinterpret_cast<float4*>(out.data_ptr<float>()), 
            size4);
    } else {
        const int block_size = 256;
        const int num_blocks = (size + block_size - 1) / block_size;
        
        fused_gelu_kernel<<<num_blocks, block_size>>>(
            x.data_ptr<float>(), out.data_ptr<float>(), size);
    }
    
    return out;
}

// Optimized LayerNorm with larger block size for better reduction
__global__ void fused_layernorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int batch_seq,
    int hidden_dim,
    float eps
) {
    extern __shared__ float shared[];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    if (row >= batch_seq) return;
    
    const float* row_x = x + row * hidden_dim;
    float* row_out = out + row * hidden_dim;
    
    // First pass: compute sum and sum of squares
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    for (int i = tid; i < hidden_dim; i += block_size) {
        float val = row_x[i];
        sum += val;
        sum_sq += val * val;
    }
    
    // Warp reduction
    for (int offset = 32; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset);
        sum_sq += __shfl_down(sum_sq, offset);
    }
    
    // Store per-warp results
    int warp_id = tid / 64;
    int lane_id = tid % 64;
    int num_warps = (block_size + 63) / 64;
    
    float* shared_sum = shared;
    float* shared_sum_sq = shared + num_warps;
    
    if (lane_id == 0) {
        shared_sum[warp_id] = sum;
        shared_sum_sq[warp_id] = sum_sq;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (tid < num_warps) {
        sum = shared_sum[tid];
        sum_sq = shared_sum_sq[tid];
    } else {
        sum = 0.0f;
        sum_sq = 0.0f;
    }
    
    if (warp_id == 0) {
        for (int offset = 32; offset > 0; offset /= 2) {
            sum += __shfl_down(sum, offset);
            sum_sq += __shfl_down(sum_sq, offset);
        }
        
        if (tid == 0) {
            float mean = sum / hidden_dim;
            float var = sum_sq / hidden_dim - mean * mean;
            shared[0] = mean;
            shared[1] = rsqrtf(var + eps);
        }
    }
    __syncthreads();
    
    float mean = shared[0];
    float inv_std = shared[1];
    
    // Second pass: normalize and scale
    for (int i = tid; i < hidden_dim; i += block_size) {
        float val = row_x[i];
        row_out[i] = (val - mean) * inv_std * gamma[i] + beta[i];
    }
}

torch::Tensor fused_layernorm_hip(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float eps) {
    auto sizes = x.sizes();
    int batch_seq = 1;
    for (size_t i = 0; i < sizes.size() - 1; i++) {
        batch_seq *= sizes[i];
    }
    int hidden_dim = sizes[sizes.size() - 1];
    
    auto out = torch::empty_like(x);
    
    // Use larger block size for better reduction performance
    int block_size = 512;
    int num_warps = (block_size + 63) / 64;
    int shared_mem = (num_warps * 2 + 2) * sizeof(float);
    
    fused_layernorm_kernel<<<batch_seq, block_size, shared_mem>>>(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_seq,
        hidden_dim,
        eps
    );
    
    return out;
}

// Fused scale, causal mask, and softmax kernel
__global__ void fused_scale_mask_softmax_kernel(
    const float* __restrict__ att,
    const float* __restrict__ mask,
    float* __restrict__ out,
    float scale,
    int B, int nh, int T
) {
    // Each block handles one row in the attention matrix
    extern __shared__ float sdata[];
    
    int batch_head = blockIdx.x / T;  // which (batch, head) pair
    int row = blockIdx.x % T;         // which row in the T x T matrix
    int tid = threadIdx.x;
    
    int b = batch_head / nh;
    int h = batch_head % nh;
    
    const float* att_row = att + (b * nh + h) * T * T + row * T;
    float* out_row = out + (b * nh + h) * T * T + row * T;
    
    // Load, scale and mask - store in shared memory
    float val = -FLT_MAX;
    if (tid <= row && tid < T) {
        val = att_row[tid] * scale;
    }
    
    // Find max for numerical stability
    float max_val = val;
    for (int offset = 32; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down(max_val, offset));
    }
    
    // Store warp maxes
    int warp_id = tid / 64;
    int lane_id = tid % 64;
    int num_warps = (blockDim.x + 63) / 64;
    
    if (lane_id == 0) {
        sdata[warp_id] = max_val;
    }
    __syncthreads();
    
    if (tid < num_warps) {
        max_val = sdata[tid];
    } else {
        max_val = -FLT_MAX;
    }
    
    if (warp_id == 0) {
        for (int offset = 32; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_down(max_val, offset));
        }
        if (tid == 0) {
            sdata[0] = max_val;
        }
    }
    __syncthreads();
    max_val = sdata[0];
    
    // Compute exp and sum
    float exp_val = 0.0f;
    if (tid <= row && tid < T) {
        exp_val = expf(val - max_val);
    }
    
    float sum_exp = exp_val;
    for (int offset = 32; offset > 0; offset /= 2) {
        sum_exp += __shfl_down(sum_exp, offset);
    }
    
    if (lane_id == 0) {
        sdata[warp_id] = sum_exp;
    }
    __syncthreads();
    
    if (tid < num_warps) {
        sum_exp = sdata[tid];
    } else {
        sum_exp = 0.0f;
    }
    
    if (warp_id == 0) {
        for (int offset = 32; offset > 0; offset /= 2) {
            sum_exp += __shfl_down(sum_exp, offset);
        }
        if (tid == 0) {
            sdata[0] = sum_exp;
        }
    }
    __syncthreads();
    sum_exp = sdata[0];
    
    // Write output
    if (tid < T) {
        if (tid <= row) {
            out_row[tid] = exp_val / sum_exp;
        } else {
            out_row[tid] = 0.0f;
        }
    }
}

torch::Tensor fused_scale_mask_softmax_hip(torch::Tensor att, torch::Tensor mask, float scale) {
    auto sizes = att.sizes();
    int B = sizes[0];
    int nh = sizes[1];
    int T = sizes[2];
    
    auto out = torch::empty_like(att);
    
    int block_size = 1024;
    while (block_size > T) block_size /= 2;
    if (block_size < 64) block_size = 64;
    
    int num_blocks = B * nh * T;
    int num_warps = (block_size + 63) / 64;
    int shared_mem = num_warps * sizeof(float);
    
    fused_scale_mask_softmax_kernel<<<num_blocks, block_size, shared_mem>>>(
        att.data_ptr<float>(),
        mask.data_ptr<float>(),
        out.data_ptr<float>(),
        scale,
        B, nh, T
    );
    
    return out;
}
"""

fused_ops = load_inline(
    name="fused_ops_v2",
    cpp_sources=cpp_source,
    cuda_sources=hip_source,
    functions=["fused_gelu_hip", "fused_layernorm_hip", "fused_scale_mask_softmax_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class NewGELU(nn.Module):
    def __init__(self):
        super(NewGELU, self).__init__()
    
    def forward(self, x):
        return fused_ops.fused_gelu_hip(x)


class FusedLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x):
        return fused_ops.fused_layernorm_hip(x, self.weight, self.bias, self.eps)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd
        self.scale = 1.0 / math.sqrt(n_embd // n_head)

    def forward(self, x):
        B, T, C = x.size()
        
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Compute attention scores
        att = q @ k.transpose(-2, -1)
        
        # Use fused scale, mask, and softmax
        att = fused_ops.fused_scale_mask_softmax_hip(att, self.bias[:,:,:T,:T], self.scale)
        
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y


class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = FusedLayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = FusedLayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


def custom_kernel(inputs):
    x = inputs[0]
    n_embd = 768
    n_head = 8
    attn_pdrop = 0.0
    resid_pdrop = 0.0
    max_seqlen = 1024
    
    model = ModelNew(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen).cuda()
    model.eval()
    
    with torch.no_grad():
        return model(x)
