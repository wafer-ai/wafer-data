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

// Warp reduce sum using warp shuffles
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Block reduce sum using shared memory
template<int BLOCK_SIZE>
__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];
    
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warpReduceSum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    int numWarps = BLOCK_SIZE / WARP_SIZE;
    val = (threadIdx.x < numWarps) ? shared[threadIdx.x] : 0.0f;
    
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

// Vectorized load - load 4 floats at once
__device__ __forceinline__ float4 load_float4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

// Vectorized store
__device__ __forceinline__ void store_float4(float* ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
}

// Optimized LayerNorm kernel with vectorized loads
// Each block processes one batch element
template<int BLOCK_SIZE>
__global__ void layernorm_kernel_vectorized(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int batch_size,
    int normalized_size,
    float eps
) {
    __shared__ float mean_shared;
    __shared__ float inv_std_shared;
    
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* x = input + batch_idx * normalized_size;
    float* y = output + batch_idx * normalized_size;
    
    int vec_size = normalized_size / 4;
    int remainder = normalized_size % 4;
    
    // Step 1: Compute sum for mean using vectorized loads
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vec_size; i += BLOCK_SIZE) {
        float4 vals = load_float4(x + i * 4);
        local_sum += vals.x + vals.y + vals.z + vals.w;
    }
    // Handle remainder
    int base = vec_size * 4;
    for (int i = threadIdx.x; i < remainder; i += BLOCK_SIZE) {
        local_sum += x[base + i];
    }
    
    float total_sum = blockReduceSum<BLOCK_SIZE>(local_sum);
    
    if (threadIdx.x == 0) {
        mean_shared = total_sum / (float)normalized_size;
    }
    __syncthreads();
    
    float mean = mean_shared;
    
    // Step 2: Compute variance using vectorized loads
    float local_var_sum = 0.0f;
    for (int i = threadIdx.x; i < vec_size; i += BLOCK_SIZE) {
        float4 vals = load_float4(x + i * 4);
        float d0 = vals.x - mean;
        float d1 = vals.y - mean;
        float d2 = vals.z - mean;
        float d3 = vals.w - mean;
        local_var_sum += d0*d0 + d1*d1 + d2*d2 + d3*d3;
    }
    // Handle remainder
    for (int i = threadIdx.x; i < remainder; i += BLOCK_SIZE) {
        float diff = x[base + i] - mean;
        local_var_sum += diff * diff;
    }
    
    float total_var = blockReduceSum<BLOCK_SIZE>(local_var_sum);
    
    if (threadIdx.x == 0) {
        float variance = total_var / (float)normalized_size;
        inv_std_shared = rsqrtf(variance + eps);
    }
    __syncthreads();
    
    float inv_std = inv_std_shared;
    
    // Step 3: Normalize and apply affine transformation with vectorized stores
    for (int i = threadIdx.x; i < vec_size; i += BLOCK_SIZE) {
        float4 vals = load_float4(x + i * 4);
        float4 g = load_float4(gamma + i * 4);
        float4 b = load_float4(beta + i * 4);
        
        float4 out;
        out.x = (vals.x - mean) * inv_std * g.x + b.x;
        out.y = (vals.y - mean) * inv_std * g.y + b.y;
        out.z = (vals.z - mean) * inv_std * g.z + b.z;
        out.w = (vals.w - mean) * inv_std * g.w + b.w;
        
        store_float4(y + i * 4, out);
    }
    // Handle remainder
    for (int i = threadIdx.x; i < remainder; i += BLOCK_SIZE) {
        int idx = base + i;
        float normalized = (x[idx] - mean) * inv_std;
        y[idx] = normalized * gamma[idx] + beta[idx];
    }
}

torch::Tensor layernorm_hip(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, double eps) {
    auto batch_size = input.size(0);
    auto normalized_size = input.numel() / batch_size;
    
    auto output = torch::empty_like(input);
    
    const int block_size = 1024;
    const int num_blocks = batch_size;
    
    layernorm_kernel_vectorized<1024><<<num_blocks, block_size>>>(
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
        """
        Initializes the LayerNorm layer.
        """
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = 1e-5
        
        # Initialize learnable parameters
        normalized_size = 1
        for s in normalized_shape:
            normalized_size *= s
        
        self.weight = nn.Parameter(torch.ones(normalized_size))
        self.bias = nn.Parameter(torch.zeros(normalized_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor.
        """
        x = x.contiguous()
        return layernorm_module.layernorm_hip(x, self.weight, self.bias, self.eps)


def get_inputs():
    x = torch.rand(16, 64, 256, 256).cuda()
    return [x]


def get_init_inputs():
    return [(64, 256, 256)]
