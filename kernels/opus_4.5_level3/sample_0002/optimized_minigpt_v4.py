import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized GELU and LayerNorm kernels with better memory coalescing
hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define WARP_SIZE 64

// Vectorized GELU kernel
__global__ void fused_gelu_kernel_vec4(const float4* __restrict__ input, 
                                        float4* __restrict__ output, 
                                        int size4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size4) {
        float4 in = __ldg(&input[idx]);
        const float sqrt_2_over_pi = 0.7978845608028654f;
        const float coef = 0.044715f;
        
        float4 out;
        
        float x = in.x;
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coef * x3);
        out.x = 0.5f * x * (1.0f + tanhf(inner));
        
        x = in.y;
        x3 = x * x * x;
        inner = sqrt_2_over_pi * (x + coef * x3);
        out.y = 0.5f * x * (1.0f + tanhf(inner));
        
        x = in.z;
        x3 = x * x * x;
        inner = sqrt_2_over_pi * (x + coef * x3);
        out.z = 0.5f * x * (1.0f + tanhf(inner));
        
        x = in.w;
        x3 = x * x * x;
        inner = sqrt_2_over_pi * (x + coef * x3);
        out.w = 0.5f * x * (1.0f + tanhf(inner));
        
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

// Optimized LayerNorm with one-pass Welford and vectorized loads
__global__ void layernorm_kernel_onepass(const float* __restrict__ input,
                                          const float* __restrict__ gamma,
                                          const float* __restrict__ beta,
                                          float* __restrict__ output,
                                          int batch_size,
                                          int hidden_size,
                                          float eps) {
    __shared__ float s_mean;
    __shared__ float s_inv_std;
    __shared__ float s_sum[16];
    __shared__ float s_sum2[16];
    
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    const float* row_input = input + row * hidden_size;
    float* row_output = output + row * hidden_size;
    
    // Compute sum and sum of squares in one pass
    float local_sum = 0.0f;
    float local_sum2 = 0.0f;
    
    // Process 4 elements at a time if possible
    int vec_end = (hidden_size / 4) * 4;
    for (int i = threadIdx.x * 4; i < vec_end; i += blockDim.x * 4) {
        float4 vals = *reinterpret_cast<const float4*>(row_input + i);
        local_sum += vals.x + vals.y + vals.z + vals.w;
        local_sum2 += vals.x * vals.x + vals.y * vals.y + vals.z * vals.z + vals.w * vals.w;
    }
    // Handle remainder
    for (int i = vec_end + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = row_input[i];
        local_sum += val;
        local_sum2 += val * val;
    }
    
    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        local_sum += __shfl_down(local_sum, offset);
        local_sum2 += __shfl_down(local_sum2, offset);
    }
    
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    if (lane == 0) {
        s_sum[wid] = local_sum;
        s_sum2[wid] = local_sum2;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (wid == 0) {
        local_sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? s_sum[threadIdx.x] : 0.0f;
        local_sum2 = (threadIdx.x < blockDim.x / WARP_SIZE) ? s_sum2[threadIdx.x] : 0.0f;
        
        #pragma unroll
        for (int offset = 8; offset > 0; offset /= 2) {
            local_sum += __shfl_down(local_sum, offset);
            local_sum2 += __shfl_down(local_sum2, offset);
        }
        
        if (threadIdx.x == 0) {
            float mean = local_sum / hidden_size;
            float variance = local_sum2 / hidden_size - mean * mean;
            s_mean = mean;
            s_inv_std = rsqrtf(variance + eps);
        }
    }
    __syncthreads();
    
    float mean = s_mean;
    float inv_std = s_inv_std;
    
    // Normalize and scale with vectorized operations
    for (int i = threadIdx.x * 4; i < vec_end; i += blockDim.x * 4) {
        float4 vals = *reinterpret_cast<const float4*>(row_input + i);
        float4 g = *reinterpret_cast<const float4*>(gamma + i);
        float4 b = *reinterpret_cast<const float4*>(beta + i);
        
        float4 result;
        result.x = g.x * (vals.x - mean) * inv_std + b.x;
        result.y = g.y * (vals.y - mean) * inv_std + b.y;
        result.z = g.z * (vals.z - mean) * inv_std + b.z;
        result.w = g.w * (vals.w - mean) * inv_std + b.w;
        
        *reinterpret_cast<float4*>(row_output + i) = result;
    }
    // Handle remainder
    for (int i = vec_end + threadIdx.x; i < hidden_size; i += blockDim.x) {
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
    
    layernorm_kernel_onepass<<<batch_size, block_size>>>(
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

// Fused residual add
__global__ void residual_add_kernel(float* __restrict__ output,
                                     const float* __restrict__ residual,
                                     int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] += residual[idx];
    }
}

torch::Tensor residual_add(torch::Tensor x, torch::Tensor residual) {
    int size = x.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    residual_add_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        residual.data_ptr<float>(),
        size
    );
    
    return x;
}
"""

custom_ops = load_inline(
    name="custom_ops_v4",
    cpp_sources=hip_source,
    functions=["fused_gelu", "fused_layernorm", "residual_add"],
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
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        # Attention block with residual
        x = x + self.attn(self.ln_1(x))
        
        # MLP block with fused GELU
        h = self.ln_2(x)
        h = self.c_fc(h)
        h = custom_ops.fused_gelu(h)
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
