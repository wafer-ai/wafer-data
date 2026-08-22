import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Kernels for GELU, LayerNorm, and fused residual add
hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define WARP_SIZE 64

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// GELU with float4 vectorization
__global__ void gelu_vec4_kernel(const float4* __restrict__ input, 
                                  float4* __restrict__ output, 
                                  int n) {
    const float c1 = 0.7978845608028654f;
    const float c2 = 0.044715f;
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float4 v = input[i];
        
        float x = v.x; v.x = 0.5f * x * (1.0f + tanhf(c1 * (x + c2 * x * x * x)));
        x = v.y; v.y = 0.5f * x * (1.0f + tanhf(c1 * (x + c2 * x * x * x)));
        x = v.z; v.z = 0.5f * x * (1.0f + tanhf(c1 * (x + c2 * x * x * x)));
        x = v.w; v.w = 0.5f * x * (1.0f + tanhf(c1 * (x + c2 * x * x * x)));
        
        output[i] = v;
    }
}

torch::Tensor fused_gelu(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    int size4 = size / 4;
    
    const int block = 256;
    const int grid = min((size4 + block - 1) / block, 4096);
    
    gelu_vec4_kernel<<<grid, block>>>(
        reinterpret_cast<const float4*>(input.data_ptr<float>()),
        reinterpret_cast<float4*>(output.data_ptr<float>()),
        size4
    );
    
    return output;
}

// LayerNorm kernel
__global__ void layernorm_kernel(const float* __restrict__ input,
                                  const float* __restrict__ gamma,
                                  const float* __restrict__ beta,
                                  float* __restrict__ output,
                                  int num_rows,
                                  int hidden_size,
                                  float eps) {
    const int row = blockIdx.x;
    if (row >= num_rows) return;
    
    const float* x = input + row * hidden_size;
    float* y = output + row * hidden_size;
    
    float sum = 0.0f, sum_sq = 0.0f;
    
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = x[i];
        sum += val;
        sum_sq += val * val;
    }
    
    // Warp reduce
    sum = warp_reduce_sum(sum);
    sum_sq = warp_reduce_sum(sum_sq);
    
    __shared__ float s_sum[4], s_sum_sq[4];
    int wid = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    
    if (lane == 0) {
        s_sum[wid] = sum;
        s_sum_sq[wid] = sum_sq;
    }
    __syncthreads();
    
    if (wid == 0) {
        sum = (lane < 4) ? s_sum[lane] : 0.0f;
        sum_sq = (lane < 4) ? s_sum_sq[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        sum_sq = warp_reduce_sum(sum_sq);
    }
    
    __shared__ float s_mean, s_rstd;
    if (threadIdx.x == 0) {
        float mean = sum / hidden_size;
        float var = sum_sq / hidden_size - mean * mean;
        s_mean = mean;
        s_rstd = rsqrtf(var + eps);
    }
    __syncthreads();
    
    float mean = s_mean, rstd = s_rstd;
    
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        y[i] = (x[i] - mean) * rstd * gamma[i] + beta[i];
    }
}

torch::Tensor fused_layernorm(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps) {
    auto sizes = input.sizes();
    int num_rows = 1;
    for (size_t i = 0; i < sizes.size() - 1; i++) {
        num_rows *= sizes[i];
    }
    int hidden_size = sizes[sizes.size() - 1];
    
    auto output = torch::empty_like(input);
    
    layernorm_kernel<<<num_rows, 256>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        num_rows,
        hidden_size,
        eps
    );
    
    return output;
}

// Fused residual add with vectorization
__global__ void residual_add_vec4_kernel(const float4* __restrict__ x,
                                          const float4* __restrict__ residual,
                                          float4* __restrict__ output,
                                          int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float4 a = x[i];
        float4 b = residual[i];
        output[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
}

torch::Tensor residual_add(torch::Tensor x, torch::Tensor residual) {
    auto output = torch::empty_like(x);
    int size = x.numel();
    int size4 = size / 4;
    
    const int block = 256;
    const int grid = min((size4 + block - 1) / block, 4096);
    
    residual_add_vec4_kernel<<<grid, block>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<const float4*>(residual.data_ptr<float>()),
        reinterpret_cast<float4*>(output.data_ptr<float>()),
        size4
    );
    
    return output;
}
"""

custom_ops = load_inline(
    name="custom_ops_v8",
    cpp_sources=hip_source,
    functions=["fused_gelu", "fused_layernorm", "residual_add"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class FusedLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x):
        return custom_ops.fused_layernorm(x.contiguous(), self.weight, self.bias, self.eps)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

    def forward(self, x):
        B, T, C = x.size()
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = FusedLayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = FusedLayerNorm(n_embd)
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        # Attention block
        attn_out = self.attn(self.ln_1(x))
        x = custom_ops.residual_add(attn_out, x)
        
        # MLP block
        h = self.ln_2(x)
        h = self.c_fc(h)
        h = custom_ops.fused_gelu(h)
        h = self.c_proj(h)
        h = self.dropout(h)
        x = custom_ops.residual_add(h, x)
        
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
