import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

layernorm_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define WARP_SIZE 64
#define BLOCK_SIZE 512

// Warp reduce sum using warp shuffles
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Warp reduce to get both sum and sum of squares
__device__ __forceinline__ void warpReduceSumSumSq(float& sum, float& sumSq) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset);
        sumSq += __shfl_down(sumSq, offset);
    }
}

// Two-pass LayerNorm with optimized memory access
__global__ void layernorm_kernel_optimized(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int batch_size,
    int normalized_size,
    float eps
) {
    // Shared memory for warp partial sums
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float s_sum[NUM_WARPS];
    __shared__ float s_sumSq[NUM_WARPS];
    __shared__ float s_mean;
    __shared__ float s_inv_std;
    
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* x = input + batch_idx * normalized_size;
    float* y = output + batch_idx * normalized_size;
    
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    
    // Each thread processes multiple elements
    // Use vectorized loads (4 floats at a time)
    int vec_size = normalized_size / 4;
    
    float local_sum = 0.0f;
    float local_sumSq = 0.0f;
    
    // Strided loop for better memory coalescing
    for (int i = tid; i < vec_size; i += BLOCK_SIZE) {
        float4 vals = *reinterpret_cast<const float4*>(x + i * 4);
        
        local_sum += vals.x + vals.y + vals.z + vals.w;
        local_sumSq += vals.x * vals.x + vals.y * vals.y + vals.z * vals.z + vals.w * vals.w;
    }
    
    // Handle remainder (normalized_size % 4)
    int base = vec_size * 4;
    for (int i = base + tid; i < normalized_size; i += BLOCK_SIZE) {
        float v = x[i];
        local_sum += v;
        local_sumSq += v * v;
    }
    
    // Warp-level reduction
    warpReduceSumSumSq(local_sum, local_sumSq);
    
    // Store warp results in shared memory
    if (lane == 0) {
        s_sum[wid] = local_sum;
        s_sumSq[wid] = local_sumSq;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (wid == 0) {
        local_sum = (lane < NUM_WARPS) ? s_sum[lane] : 0.0f;
        local_sumSq = (lane < NUM_WARPS) ? s_sumSq[lane] : 0.0f;
        
        warpReduceSumSumSq(local_sum, local_sumSq);
        
        if (lane == 0) {
            float mean = local_sum / (float)normalized_size;
            // Var(X) = E[X^2] - E[X]^2
            float variance = local_sumSq / (float)normalized_size - mean * mean;
            s_mean = mean;
            s_inv_std = rsqrtf(variance + eps);
        }
    }
    __syncthreads();
    
    float mean = s_mean;
    float inv_std = s_inv_std;
    
    // Pass 2: Normalize and apply affine transformation
    for (int i = tid; i < vec_size; i += BLOCK_SIZE) {
        float4 vals = *reinterpret_cast<const float4*>(x + i * 4);
        float4 g = *reinterpret_cast<const float4*>(gamma + i * 4);
        float4 b = *reinterpret_cast<const float4*>(beta + i * 4);
        
        float4 out;
        out.x = (vals.x - mean) * inv_std * g.x + b.x;
        out.y = (vals.y - mean) * inv_std * g.y + b.y;
        out.z = (vals.z - mean) * inv_std * g.z + b.z;
        out.w = (vals.w - mean) * inv_std * g.w + b.w;
        
        *reinterpret_cast<float4*>(y + i * 4) = out;
    }
    
    // Handle remainder
    for (int i = base + tid; i < normalized_size; i += BLOCK_SIZE) {
        float normalized = (x[i] - mean) * inv_std;
        y[i] = normalized * gamma[i] + beta[i];
    }
}

torch::Tensor layernorm_hip(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, double eps) {
    auto batch_size = input.size(0);
    auto normalized_size = input.numel() / batch_size;
    
    auto output = torch::empty_like(input);
    
    layernorm_kernel_optimized<<<batch_size, BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        normalized_size,
        (float)eps
    );
    
    return output;
}
"""

layernorm_cpp_source = """
torch::Tensor layernorm_hip(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, double eps);
"""

layernorm_module = load_inline(
    name="layernorm_hip",
    cpp_sources=layernorm_cpp_source,
    cuda_sources=layernorm_hip_source,
    functions=["layernorm_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = 1e-5
        
        normalized_size = 1
        for s in normalized_shape:
            normalized_size *= s
        
        self.weight = nn.Parameter(torch.ones(normalized_size))
        self.bias = nn.Parameter(torch.zeros(normalized_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        return layernorm_module.layernorm_hip(x, self.weight, self.bias, self.eps)


def get_inputs():
    x = torch.rand(16, 64, 256, 256).cuda()
    return [x]


def get_init_inputs():
    return [(64, 256, 256)]
