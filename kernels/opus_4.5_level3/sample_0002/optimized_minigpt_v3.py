import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized GELU and LayerNorm kernels
hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define WARP_SIZE 64

// Fast tanh approximation
__device__ __forceinline__ float fast_tanh(float x) {
    float x2 = x * x;
    float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
    float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    return a / b;
}

// Vectorized GELU kernel with fast tanh
__global__ void fused_gelu_kernel_vec4(const float4* __restrict__ input, 
                                        float4* __restrict__ output, 
                                        int size4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size4) {
        float4 in = input[idx];
        const float sqrt_2_over_pi = 0.7978845608028654f;
        const float coef = 0.044715f;
        
        float4 out;
        
        #define GELU_ELEM(elem) { \
            float x = in.elem; \
            float x3 = x * x * x; \
            float inner = sqrt_2_over_pi * (x + coef * x3); \
            out.elem = 0.5f * x * (1.0f + tanhf(inner)); \
        }
        
        GELU_ELEM(x)
        GELU_ELEM(y)
        GELU_ELEM(z)
        GELU_ELEM(w)
        
        #undef GELU_ELEM
        
        output[idx] = out;
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
    
    if (size % 4 == 0 && ((uintptr_t)input.data_ptr<float>() % 16 == 0)) {
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

// Optimized LayerNorm using two-pass algorithm
// Uses shared memory and warp shuffles for reductions
__global__ void layernorm_kernel_opt(const float* __restrict__ input,
                                      const float* __restrict__ gamma,
                                      const float* __restrict__ beta,
                                      float* __restrict__ output,
                                      int batch_size,
                                      int hidden_size,
                                      float eps) {
    __shared__ float s_mean;
    __shared__ float s_inv_std;
    __shared__ float s_sum[16];  // For warp reduction
    
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    const float* row_input = input + row * hidden_size;
    float* row_output = output + row * hidden_size;
    
    // Pass 1: Compute mean using parallel reduction
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        local_sum += row_input[i];
    }
    
    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        local_sum += __shfl_down(local_sum, offset);
    }
    
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    if (lane == 0) {
        s_sum[wid] = local_sum;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (wid == 0) {
        local_sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? s_sum[threadIdx.x] : 0.0f;
        #pragma unroll
        for (int offset = 8; offset > 0; offset /= 2) {
            local_sum += __shfl_down(local_sum, offset);
        }
        if (threadIdx.x == 0) {
            s_mean = local_sum / hidden_size;
        }
    }
    __syncthreads();
    
    float mean = s_mean;
    
    // Pass 2: Compute variance
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = row_input[i] - mean;
        local_var += diff * diff;
    }
    
    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        local_var += __shfl_down(local_var, offset);
    }
    
    if (lane == 0) {
        s_sum[wid] = local_var;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (wid == 0) {
        local_var = (threadIdx.x < blockDim.x / WARP_SIZE) ? s_sum[threadIdx.x] : 0.0f;
        #pragma unroll
        for (int offset = 8; offset > 0; offset /= 2) {
            local_var += __shfl_down(local_var, offset);
        }
        if (threadIdx.x == 0) {
            s_inv_std = rsqrtf(local_var / hidden_size + eps);
        }
    }
    __syncthreads();
    
    float inv_std = s_inv_std;
    
    // Pass 3: Normalize and scale
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
    
    int block_size = 512;
    
    layernorm_kernel_opt<<<batch_size, block_size>>>(
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
    name="custom_ops_v3",
    cpp_sources=hip_source,
    functions=["fused_gelu", "fused_layernorm"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
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
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

    def forward(self, x):
        B, T, C = x.size()
        
        # Compute q, k, v projections
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for attention
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Use PyTorch's optimized scaled_dot_product_attention with causal mask
        # This uses Flash Attention on supported hardware
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
