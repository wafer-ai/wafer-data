import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized GELU + LayerNorm kernels with better memory access patterns
hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define WARP_SIZE 64

// Warp reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Fused GELU with vectorized loads
__global__ void fused_gelu_kernel_vec4(const float4* __restrict__ input, 
                                        float4* __restrict__ output, 
                                        int size4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size4) {
        float4 in = input[idx];
        const float sqrt_2_over_pi = 0.7978845608028654f;
        
        float x = in.x;
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
        float out_x = 0.5f * x * (1.0f + tanhf(inner));
        
        x = in.y;
        x3 = x * x * x;
        inner = sqrt_2_over_pi * (x + 0.044715f * x3);
        float out_y = 0.5f * x * (1.0f + tanhf(inner));
        
        x = in.z;
        x3 = x * x * x;
        inner = sqrt_2_over_pi * (x + 0.044715f * x3);
        float out_z = 0.5f * x * (1.0f + tanhf(inner));
        
        x = in.w;
        x3 = x * x * x;
        inner = sqrt_2_over_pi * (x + 0.044715f * x3);
        float out_w = 0.5f * x * (1.0f + tanhf(inner));
        
        output[idx] = make_float4(out_x, out_y, out_z, out_w);
    }
}

__global__ void fused_gelu_kernel(const float* __restrict__ input, 
                                   float* __restrict__ output, 
                                   int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        const float sqrt_2_over_pi = 0.7978845608028654f;
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

torch::Tensor fused_gelu(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    
    if (size % 4 == 0) {
        int size4 = size / 4;
        const int block_size = 256;
        const int num_blocks = (size4 + block_size - 1) / block_size;
        fused_gelu_kernel_vec4<<<num_blocks, block_size>>>(
            reinterpret_cast<const float4*>(input.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            size4
        );
    } else {
        const int block_size = 256;
        const int num_blocks = (size + block_size - 1) / block_size;
        fused_gelu_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            size
        );
    }
    
    return output;
}

// Optimized LayerNorm with Welford's algorithm for numerical stability
__global__ void layernorm_kernel_welford(const float* __restrict__ input,
                                          const float* __restrict__ gamma,
                                          const float* __restrict__ beta,
                                          float* __restrict__ output,
                                          int batch_size,
                                          int hidden_size,
                                          float eps) {
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    const float* row_input = input + row * hidden_size;
    float* row_output = output + row * hidden_size;
    
    // Welford's online algorithm for mean and variance
    float mean = 0.0f;
    float m2 = 0.0f;
    int count = 0;
    
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = row_input[i];
        count++;
        float delta = val - mean;
        mean += delta / count;
        float delta2 = val - mean;
        m2 += delta * delta2;
    }
    
    // Parallel reduction within warp
    __shared__ float shared_mean[16];
    __shared__ float shared_m2[16];
    __shared__ int shared_count[16];
    
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    // Warp-level reduction using parallel Welford
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        float other_mean = __shfl_down(mean, offset);
        float other_m2 = __shfl_down(m2, offset);
        int other_count = __shfl_down(count, offset);
        
        if (count + other_count > 0) {
            int new_count = count + other_count;
            float delta = other_mean - mean;
            float new_mean = mean + delta * other_count / new_count;
            float new_m2 = m2 + other_m2 + delta * delta * count * other_count / new_count;
            mean = new_mean;
            m2 = new_m2;
            count = new_count;
        }
    }
    
    if (lane == 0) {
        shared_mean[wid] = mean;
        shared_m2[wid] = m2;
        shared_count[wid] = count;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (threadIdx.x < blockDim.x / WARP_SIZE) {
        mean = shared_mean[threadIdx.x];
        m2 = shared_m2[threadIdx.x];
        count = shared_count[threadIdx.x];
    } else {
        mean = 0.0f;
        m2 = 0.0f;
        count = 0;
    }
    
    if (wid == 0) {
        #pragma unroll
        for (int offset = blockDim.x / WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_mean = __shfl_down(mean, offset);
            float other_m2 = __shfl_down(m2, offset);
            int other_count = __shfl_down(count, offset);
            
            if (count + other_count > 0) {
                int new_count = count + other_count;
                float delta = other_mean - mean;
                float new_mean = mean + delta * other_count / new_count;
                float new_m2 = m2 + other_m2 + delta * delta * count * other_count / new_count;
                mean = new_mean;
                m2 = new_m2;
                count = new_count;
            }
        }
    }
    
    __shared__ float final_mean;
    __shared__ float final_inv_std;
    
    if (threadIdx.x == 0) {
        final_mean = mean;
        float variance = m2 / hidden_size;
        final_inv_std = rsqrtf(variance + eps);
    }
    __syncthreads();
    
    // Normalize and apply scale/shift
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = (row_input[i] - final_mean) * final_inv_std;
        row_output[i] = gamma[i] * normalized + beta[i];
    }
}

torch::Tensor fused_layernorm(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps) {
    auto sizes = input.sizes();
    int batch_size = 1;
    for (size_t i = 0; i < sizes.size() - 1; i++) {
        batch_size *= sizes[i];
    }
    int hidden_size = sizes[sizes.size() - 1];
    
    auto output = torch::empty_like(input);
    
    int block_size = 256;
    if (hidden_size > 512) block_size = 512;
    
    layernorm_kernel_welford<<<batch_size, block_size>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        hidden_size,
        eps
    );
    
    return output;
}
"""

custom_ops = load_inline(
    name="custom_ops_v2",
    cpp_sources=hip_source,
    functions=["fused_gelu", "fused_layernorm"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class FusedGELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return custom_ops.fused_gelu(x)


class FusedLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x):
        return custom_ops.fused_layernorm(x, self.weight, self.bias, self.eps)


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

    def forward(self, x):
        B, T, C = x.size()
        
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
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
            act     = FusedGELU(),
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
