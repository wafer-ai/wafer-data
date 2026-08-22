import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused dropout + softmax kernel with vectorized loads (float4)
fused_dropout_softmax_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hiprand/hiprand_kernel.h>
#include <cmath>

#define WARP_SIZE 64
#define BLOCK_SIZE 512

// Warp reduce max
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

// Warp reduce sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Block reduce max
__device__ __forceinline__ float block_reduce_max(float val) {
    const int num_warps = BLOCK_SIZE / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];
    
    val = warp_reduce_max(val);
    if (lane_id == 0) shared[warp_id] = val;
    __syncthreads();
    
    if (warp_id == 0) {
        val = (lane_id < num_warps) ? shared[lane_id] : -INFINITY;
        val = warp_reduce_max(val);
    }
    __syncthreads();
    return __shfl(val, 0);
}

// Block reduce sum
__device__ __forceinline__ float block_reduce_sum(float val) {
    const int num_warps = BLOCK_SIZE / WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];
    
    val = warp_reduce_sum(val);
    if (lane_id == 0) shared[warp_id] = val;
    __syncthreads();
    
    if (warp_id == 0) {
        val = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
    }
    __syncthreads();
    return __shfl(val, 0);
}

__global__ void fused_dropout_softmax_vec_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int rows,
    const int cols,
    const float dropout_p,
    const float scale,
    const unsigned long long seed,
    const bool training
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    
    const int tid = threadIdx.x;
    
    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;
    
    // Initialize random state
    hiprandStatePhilox4_32_10_t state;
    if (training) {
        hiprand_init(seed, (unsigned long long)(row * cols + tid * 4), 0, &state);
    }
    
    // Use float4 for vectorized access where possible
    const int cols_vec = cols / 4;
    const float4* input_vec = reinterpret_cast<const float4*>(row_input);
    float4* output_vec = reinterpret_cast<float4*>(row_output);
    
    // Phase 1: Find max
    float local_max = -INFINITY;
    
    for (int i = tid; i < cols_vec; i += BLOCK_SIZE) {
        float4 val = input_vec[i];
        
        if (training) {
            float4 rand_vals = hiprand_uniform4(&state);
            val.x = (rand_vals.x < dropout_p) ? -INFINITY : val.x * scale;
            val.y = (rand_vals.y < dropout_p) ? -INFINITY : val.y * scale;
            val.z = (rand_vals.z < dropout_p) ? -INFINITY : val.z * scale;
            val.w = (rand_vals.w < dropout_p) ? -INFINITY : val.w * scale;
        }
        
        local_max = fmaxf(local_max, fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w)));
    }
    
    float global_max = block_reduce_max(local_max);
    
    // Phase 2: Sum of exp
    // Re-init RNG state
    if (training) {
        hiprand_init(seed, (unsigned long long)(row * cols + tid * 4), 0, &state);
    }
    
    float local_sum = 0.0f;
    
    for (int i = tid; i < cols_vec; i += BLOCK_SIZE) {
        float4 val = input_vec[i];
        
        if (training) {
            float4 rand_vals = hiprand_uniform4(&state);
            val.x = (rand_vals.x < dropout_p) ? -INFINITY : val.x * scale;
            val.y = (rand_vals.y < dropout_p) ? -INFINITY : val.y * scale;
            val.z = (rand_vals.z < dropout_p) ? -INFINITY : val.z * scale;
            val.w = (rand_vals.w < dropout_p) ? -INFINITY : val.w * scale;
        }
        
        local_sum += expf(val.x - global_max);
        local_sum += expf(val.y - global_max);
        local_sum += expf(val.z - global_max);
        local_sum += expf(val.w - global_max);
    }
    
    float global_sum = block_reduce_sum(local_sum);
    float inv_sum = 1.0f / global_sum;
    
    // Phase 3: Compute softmax
    // Re-init RNG state
    if (training) {
        hiprand_init(seed, (unsigned long long)(row * cols + tid * 4), 0, &state);
    }
    
    for (int i = tid; i < cols_vec; i += BLOCK_SIZE) {
        float4 val = input_vec[i];
        
        if (training) {
            float4 rand_vals = hiprand_uniform4(&state);
            val.x = (rand_vals.x < dropout_p) ? -INFINITY : val.x * scale;
            val.y = (rand_vals.y < dropout_p) ? -INFINITY : val.y * scale;
            val.z = (rand_vals.z < dropout_p) ? -INFINITY : val.z * scale;
            val.w = (rand_vals.w < dropout_p) ? -INFINITY : val.w * scale;
        }
        
        float4 result;
        result.x = expf(val.x - global_max) * inv_sum;
        result.y = expf(val.y - global_max) * inv_sum;
        result.z = expf(val.z - global_max) * inv_sum;
        result.w = expf(val.w - global_max) * inv_sum;
        
        output_vec[i] = result;
    }
}

torch::Tensor fused_dropout_softmax_hip(torch::Tensor input, float dropout_p, bool training) {
    auto output = torch::empty_like(input);
    
    const int rows = input.size(0);
    const int cols = input.size(1);
    
    float scale = 1.0f / (1.0f - dropout_p);
    
    unsigned long long seed = training ? (unsigned long long)(rand()) : 0;
    
    hipLaunchKernelGGL(fused_dropout_softmax_vec_kernel,
        dim3(rows), dim3(BLOCK_SIZE), 0, 0,
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols,
        dropout_p,
        scale,
        seed,
        training
    );
    
    return output;
}
"""

fused_dropout_softmax_cpp = """
torch::Tensor fused_dropout_softmax_hip(torch::Tensor input, float dropout_p, bool training);
"""

fused_module = load_inline(
    name="fused_dropout_softmax_v3",
    cpp_sources=fused_dropout_softmax_cpp,
    cuda_sources=fused_dropout_softmax_source,
    functions=["fused_dropout_softmax_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    """
    Optimized model with fused dropout + softmax kernel using vectorized memory access.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout_p = dropout_p
        self.fused_module = fused_module

    def forward(self, x):
        x = self.matmul(x)
        x = self.fused_module.fused_dropout_softmax_hip(x, self.dropout_p, self.training)
        return x
