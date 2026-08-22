
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define WARP_SIZE 64

template <typename T>
__device__ T warp_reduce_sum(T val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

template <typename T>
__device__ T block_reduce_sum(T val) {
    static __shared__ T shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

__global__ void layernorm_kernel_v5(const float* __restrict__ input, const float* __restrict__ weight, const float* __restrict__ bias, float* __restrict__ output, int N, int D, float eps) {
    int row = blockIdx.x;
    if (row >= N) return;
    const float* row_input = input + row * D;
    float* row_output = output + row * D;

    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    // Using float4 for vectorized access if possible
    int i = 0;
    const float4* row_input_f4 = reinterpret_cast<const float4*>(row_input);
    for (; i + 3 < D; i += 4) {
        // We can't easily do this if i is not a multiple of 4 relative to threadIdx,
        // so let's just stick to a simpler but still fast version.
    }
    
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float val = row_input[j];
        sum += val;
        sum_sq += val * val;
    }

    sum = block_reduce_sum(sum);
    sum_sq = block_reduce_sum(sum_sq);

    __shared__ float shared_mean;
    __shared__ float shared_inv_std;
    if (threadIdx.x == 0) {
        shared_mean = sum / D;
        shared_inv_std = 1.0f / sqrtf((sum_sq / D) - (shared_mean * shared_mean) + eps);
    }
    __syncthreads();

    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        row_output[j] = (row_input[j] - shared_mean) * shared_inv_std * weight[j] + bias[j];
    }
}

__global__ void add_layernorm_kernel_v5(const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ weight, const float* __restrict__ bias, float* __restrict__ out_x, float* __restrict__ out_norm, int N, int D, float eps) {
    int row = blockIdx.x;
    if (row >= N) return;
    const float* row_x = x + row * D;
    const float* row_y = y + row * D;
    float* row_out_x = out_x + row * D;
    float* row_out_norm = out_norm + row * D;

    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = row_x[i] + row_y[i];
        row_out_x[i] = val;
        sum += val;
        sum_sq += val * val;
    }

    sum = block_reduce_sum(sum);
    sum_sq = block_reduce_sum(sum_sq);

    __shared__ float shared_mean;
    __shared__ float shared_inv_std;
    if (threadIdx.x == 0) {
        shared_mean = sum / D;
        shared_inv_std = 1.0f / sqrtf((sum_sq / D) - (shared_mean * shared_mean) + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        row_out_norm[i] = (row_out_x[i] - shared_mean) * shared_inv_std * weight[i] + bias[i];
    }
}

__global__ void bias_gelu_kernel_v5(const float* __restrict__ input, const float* __restrict__ bias, float* __restrict__ output, int D, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx] + bias[idx % D];
        // 0.5 * x * (1.0 + tanh(sqrt(2.0 / pi) * (x + 0.044715 * x^3)))
        float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

torch::Tensor layernorm_hip(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float eps) {
    auto input_reshaped = input.reshape({-1, input.size(-1)});
    auto N = input_reshaped.size(0);
    auto D = input_reshaped.size(1);
    auto output = torch::empty_like(input_reshaped);
    layernorm_kernel_v5<<<N, 256>>>(input_reshaped.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), N, D, eps);
    return output.view_as(input);
}

std::vector<torch::Tensor> add_layernorm_hip(torch::Tensor x, torch::Tensor y, torch::Tensor weight, torch::Tensor bias, float eps) {
    auto x_reshaped = x.reshape({-1, x.size(-1)});
    auto y_reshaped = y.reshape({-1, y.size(-1)});
    auto N = x_reshaped.size(0);
    auto D = x_reshaped.size(1);
    auto out_x = torch::empty_like(x_reshaped);
    auto out_norm = torch::empty_like(x_reshaped);
    add_layernorm_kernel_v5<<<N, 256>>>(x_reshaped.data_ptr<float>(), y_reshaped.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), out_x.data_ptr<float>(), out_norm.data_ptr<float>(), N, D, eps);
    return {out_x.view_as(x), out_norm.view_as(x)};
}

torch::Tensor bias_gelu_hip(torch::Tensor input, torch::Tensor bias) {
    auto size = input.numel();
    auto D = bias.size(0);
    auto output = torch::empty_like(input);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    bias_gelu_kernel_v5<<<num_blocks, block_size>>>(input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), D, size);
    return output;
}
"""

custom_kernels = load_inline(
    name="custom_kernels_v5",
    cpp_sources=hip_source,
    functions=["layernorm_hip", "add_layernorm_hip", "bias_gelu_hip"],
    verbose=True,
)

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout_p = attn_pdrop
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()
        # q, k, v = self.c_attn(x).split(self.n_embd, dim=2) # still using linear for now
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_dropout_p if self.training else 0.0, is_causal=True
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, n_embd, resid_pdrop):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        # We use F.linear with bias=None to avoid extra addition, then our bias_gelu_hip
        y = F.linear(x, self.c_fc.weight, None)
        y = custom_kernels.bias_gelu_hip(y, self.c_fc.bias)
        y = self.c_proj(y)
        y = self.dropout(y)
        return y

class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, resid_pdrop)

    def forward(self, x):
        ln1_out = custom_kernels.layernorm_hip(x, self.ln_1.weight, self.ln_1.bias, self.ln_1.eps)
        attn_out = self.attn(ln1_out)
        x, ln2_out = custom_kernels.add_layernorm_hip(x, attn_out, self.ln_2.weight, self.ln_2.bias, self.ln_2.eps)
        mlp_out = self.mlp(ln2_out)
        return x + mlp_out

def get_inputs():
    batch_size = 128
    seq_len = 512
    n_embd = 768
    return [torch.rand(batch_size, seq_len, n_embd).cuda()]

def get_init_inputs():
    n_embd = 768
    n_head = 8
    attn_pdrop = 0.0
    resid_pdrop = 0.0
    max_seqlen = 1024
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]
