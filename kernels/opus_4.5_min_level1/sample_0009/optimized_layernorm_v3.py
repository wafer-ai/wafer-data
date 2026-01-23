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
#define BLOCK_SIZE 256
#define ELEMS_PER_THREAD 16

// Warp reduce sum using warp shuffles
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Block reduce sum 
__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];
    
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warpReduceSum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    constexpr int numWarps = BLOCK_SIZE / WARP_SIZE;
    val = (threadIdx.x < numWarps) ? shared[threadIdx.x] : 0.0f;
    
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

// Welford's online algorithm for numerically stable mean/variance
// Fused single-pass computation of mean and variance
__global__ void layernorm_kernel_welford(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int batch_size,
    int normalized_size,
    float eps
) {
    __shared__ float s_mean;
    __shared__ float s_inv_std;
    
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* x = input + batch_idx * normalized_size;
    float* y = output + batch_idx * normalized_size;
    
    // Welford's algorithm for online variance computation
    float mean = 0.0f;
    float M2 = 0.0f;
    float count = 0.0f;
    
    // Process elements with vectorized loads
    int vec_size = normalized_size / 4;
    
    #pragma unroll 4
    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        float4 vals = *reinterpret_cast<const float4*>(x + i * 4);
        
        // Process each element with Welford's update
        float delta, delta2;
        
        count += 1.0f;
        delta = vals.x - mean;
        mean += delta / count;
        delta2 = vals.x - mean;
        M2 += delta * delta2;
        
        count += 1.0f;
        delta = vals.y - mean;
        mean += delta / count;
        delta2 = vals.y - mean;
        M2 += delta * delta2;
        
        count += 1.0f;
        delta = vals.z - mean;
        mean += delta / count;
        delta2 = vals.z - mean;
        M2 += delta * delta2;
        
        count += 1.0f;
        delta = vals.w - mean;
        mean += delta / count;
        delta2 = vals.w - mean;
        M2 += delta * delta2;
    }
    
    // Handle remainder
    int base = vec_size * 4;
    for (int i = base + threadIdx.x; i < normalized_size; i += blockDim.x) {
        count += 1.0f;
        float delta = x[i] - mean;
        mean += delta / count;
        float delta2 = x[i] - mean;
        M2 += delta * delta2;
    }
    
    // Parallel reduction of Welford's partial results
    // This uses the parallel Welford merge formula
    __shared__ float s_counts[BLOCK_SIZE / WARP_SIZE];
    __shared__ float s_means[BLOCK_SIZE / WARP_SIZE];
    __shared__ float s_M2s[BLOCK_SIZE / WARP_SIZE];
    
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    // Warp-level Welford merge
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        float other_count = __shfl_down(count, offset);
        float other_mean = __shfl_down(mean, offset);
        float other_M2 = __shfl_down(M2, offset);
        
        if (other_count > 0 && count > 0) {
            float combined_count = count + other_count;
            float delta_mean = other_mean - mean;
            float new_mean = mean + delta_mean * other_count / combined_count;
            float new_M2 = M2 + other_M2 + delta_mean * delta_mean * count * other_count / combined_count;
            count = combined_count;
            mean = new_mean;
            M2 = new_M2;
        } else if (other_count > 0) {
            count = other_count;
            mean = other_mean;
            M2 = other_M2;
        }
    }
    
    if (lane == 0) {
        s_counts[wid] = count;
        s_means[wid] = mean;
        s_M2s[wid] = M2;
    }
    __syncthreads();
    
    // Final reduction in first warp
    constexpr int numWarps = BLOCK_SIZE / WARP_SIZE;
    if (wid == 0 && lane < numWarps) {
        count = s_counts[lane];
        mean = s_means[lane];
        M2 = s_M2s[lane];
        
        #pragma unroll
        for (int offset = numWarps/2; offset > 0; offset /= 2) {
            float other_count = __shfl_down(count, offset);
            float other_mean = __shfl_down(mean, offset);
            float other_M2 = __shfl_down(M2, offset);
            
            if (other_count > 0 && count > 0) {
                float combined_count = count + other_count;
                float delta_mean = other_mean - mean;
                float new_mean = mean + delta_mean * other_count / combined_count;
                float new_M2 = M2 + other_M2 + delta_mean * delta_mean * count * other_count / combined_count;
                count = combined_count;
                mean = new_mean;
                M2 = new_M2;
            } else if (other_count > 0) {
                count = other_count;
                mean = other_mean;
                M2 = other_M2;
            }
        }
        
        if (lane == 0) {
            s_mean = mean;
            float variance = M2 / (float)normalized_size;
            s_inv_std = rsqrtf(variance + eps);
        }
    }
    __syncthreads();
    
    mean = s_mean;
    float inv_std = s_inv_std;
    
    // Normalize and apply affine transformation
    #pragma unroll 4
    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
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
    for (int i = base + threadIdx.x; i < normalized_size; i += blockDim.x) {
        float normalized = (x[i] - mean) * inv_std;
        y[i] = normalized * gamma[i] + beta[i];
    }
}

torch::Tensor layernorm_hip(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, double eps) {
    auto batch_size = input.size(0);
    auto normalized_size = input.numel() / batch_size;
    
    auto output = torch::empty_like(input);
    
    layernorm_kernel_welford<<<batch_size, BLOCK_SIZE>>>(
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
    """
    Optimized model that performs Layer Normalization using custom HIP kernel.
    """
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
