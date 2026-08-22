import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized GELU and LayerNorm kernels
hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define WARP_SIZE 64
#define BLOCK_SIZE 256

// Warp reduce for float
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Block reduce using shared memory
__device__ float block_reduce_sum(float val) {
    __shared__ float shared[WARP_SIZE];
    
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[threadIdx.x] : 0.0f;
    
    if (wid == 0) val = warp_reduce_sum(val);
    
    return val;
}

// Optimized GELU using float4 vectorization
__global__ void fused_gelu_vec4_kernel(const float* __restrict__ input, 
                                        float* __restrict__ output, 
                                        int total_size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < total_size) {
        float4 in = *reinterpret_cast<const float4*>(input + idx);
        const float c = 0.7978845608028654f;  // sqrt(2/pi)
        const float k = 0.044715f;
        
        float4 out;
        
        #define GELU(v) { \
            float x = in.v; \
            float x3 = x * x * x; \
            out.v = 0.5f * x * (1.0f + tanhf(c * (x + k * x3))); \
        }
        GELU(x); GELU(y); GELU(z); GELU(w);
        #undef GELU
        
        *reinterpret_cast<float4*>(output + idx) = out;
    } else if (idx < total_size) {
        // Handle tail elements
        for (int i = idx; i < total_size && i < idx + 4; i++) {
            float x = input[i];
            const float c = 0.7978845608028654f;
            const float k = 0.044715f;
            float x3 = x * x * x;
            output[i] = 0.5f * x * (1.0f + tanhf(c * (x + k * x3)));
        }
    }
}

torch::Tensor fused_gelu(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    
    const int block_size = 256;
    const int num_elements_per_block = block_size * 4;
    const int num_blocks = (size + num_elements_per_block - 1) / num_elements_per_block;
    
    fused_gelu_vec4_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    return output;
}

// Optimized LayerNorm - one pass algorithm with better memory access
// Processes hidden_size 768 efficiently
__global__ void layernorm_kernel_768(const float* __restrict__ input,
                                      const float* __restrict__ gamma,
                                      const float* __restrict__ beta,
                                      float* __restrict__ output,
                                      int batch_size,
                                      int hidden_size,
                                      float eps) {
    __shared__ float s_gamma[768];
    __shared__ float s_beta[768];
    __shared__ float s_mean;
    __shared__ float s_inv_std;
    
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    // Load gamma and beta to shared memory
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        s_gamma[i] = gamma[i];
        s_beta[i] = beta[i];
    }
    __syncthreads();
    
    const float* row_input = input + row * hidden_size;
    float* row_output = output + row * hidden_size;
    
    // Compute sum and sum of squares in one pass
    float local_sum = 0.0f;
    float local_sum2 = 0.0f;
    
    #pragma unroll 4
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = row_input[i];
        local_sum += val;
        local_sum2 += val * val;
    }
    
    // Block-wide reduction
    float sum = block_reduce_sum(local_sum);
    __syncthreads();
    float sum2 = block_reduce_sum(local_sum2);
    
    if (threadIdx.x == 0) {
        float mean = sum / hidden_size;
        float variance = sum2 / hidden_size - mean * mean;
        s_mean = mean;
        s_inv_std = rsqrtf(variance + eps);
    }
    __syncthreads();
    
    float mean = s_mean;
    float inv_std = s_inv_std;
    
    // Normalize and scale
    #pragma unroll 4
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = (row_input[i] - mean) * inv_std;
        row_output[i] = s_gamma[i] * normalized + s_beta[i];
    }
}

// General LayerNorm kernel for any hidden size
__global__ void layernorm_kernel_general(const float* __restrict__ input,
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
    
    float local_sum = 0.0f;
    float local_sum2 = 0.0f;
    
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = row_input[i];
        local_sum += val;
        local_sum2 += val * val;
    }
    
    float sum = block_reduce_sum(local_sum);
    __syncthreads();
    float sum2 = block_reduce_sum(local_sum2);
    
    __shared__ float s_mean, s_inv_std;
    if (threadIdx.x == 0) {
        float mean = sum / hidden_size;
        float variance = sum2 / hidden_size - mean * mean;
        s_mean = mean;
        s_inv_std = rsqrtf(variance + eps);
    }
    __syncthreads();
    
    float mean = s_mean;
    float inv_std = s_inv_std;
    
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = (row_input[i] - mean) * inv_std;
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
    
    if (hidden_size == 768) {
        layernorm_kernel_768<<<batch_size, block_size>>>(
            input.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            hidden_size,
            eps
        );
    } else {
        layernorm_kernel_general<<<batch_size, block_size>>>(
            input.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            hidden_size,
            eps
        );
    }
    
    return output;
}
"""

custom_ops = load_inline(
    name="custom_ops_v5",
    cpp_sources=hip_source,
    functions=["fused_gelu", "fused_layernorm"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class FusedGELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return custom_ops.fused_gelu(x.contiguous())


class FusedLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
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
        self.attn_dropout = nn.Dropout(attn_pdrop)
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

        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True
        )
        
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
        self.gelu = FusedGELU()
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        h = self.ln_2(x)
        h = self.c_fc(h)
        h = self.gelu(h)
        h = self.c_proj(h)
        h = self.dropout(h)
        x = x + h
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
