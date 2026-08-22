import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized softmax kernel (no dropout in eval mode)
fused_softmax_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

#define WARP_SIZE 64
#define BLOCK_SIZE 1024

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

// Block reduce max with shared memory
__device__ float block_reduce_max_shared(float val) {
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
    return __shfl(val, 0);
}

// Block reduce sum with shared memory
__device__ float block_reduce_sum_shared(float val) {
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
    return __shfl(val, 0);
}

// Vectorized softmax kernel (no dropout during eval)
__global__ void softmax_vec4_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int rows,
    const int cols
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    
    const int tid = threadIdx.x;
    
    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;
    
    const int cols_vec = cols / 4;
    const float4* input_vec = reinterpret_cast<const float4*>(row_input);
    float4* output_vec = reinterpret_cast<float4*>(row_output);
    
    // Phase 1: Find max
    float local_max = -INFINITY;
    
    for (int i = tid; i < cols_vec; i += BLOCK_SIZE) {
        float4 val = input_vec[i];
        local_max = fmaxf(local_max, fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w)));
    }
    
    float global_max = block_reduce_max_shared(local_max);
    __syncthreads();
    
    // Phase 2: Sum of exp
    float local_sum = 0.0f;
    
    for (int i = tid; i < cols_vec; i += BLOCK_SIZE) {
        float4 val = input_vec[i];
        local_sum += expf(val.x - global_max);
        local_sum += expf(val.y - global_max);
        local_sum += expf(val.z - global_max);
        local_sum += expf(val.w - global_max);
    }
    
    float global_sum = block_reduce_sum_shared(local_sum);
    __syncthreads();
    float inv_sum = 1.0f / global_sum;
    
    // Phase 3: Compute softmax
    for (int i = tid; i < cols_vec; i += BLOCK_SIZE) {
        float4 val = input_vec[i];
        float4 result;
        result.x = expf(val.x - global_max) * inv_sum;
        result.y = expf(val.y - global_max) * inv_sum;
        result.z = expf(val.z - global_max) * inv_sum;
        result.w = expf(val.w - global_max) * inv_sum;
        output_vec[i] = result;
    }
}

torch::Tensor softmax_hip(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int rows = input.size(0);
    const int cols = input.size(1);
    
    hipLaunchKernelGGL(softmax_vec4_kernel,
        dim3(rows), dim3(BLOCK_SIZE), 0, 0,
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols
    );
    
    return output;
}
"""

fused_softmax_cpp = """
torch::Tensor softmax_hip(torch::Tensor input);
"""

fused_module = load_inline(
    name="fused_softmax_v4",
    cpp_sources=fused_softmax_cpp,
    cuda_sources=fused_softmax_source,
    functions=["softmax_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    """
    Optimized model - in eval mode, dropout is identity, so we just do matmul + softmax.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)
        self.dropout_p = dropout_p
        self.fused_module = fused_module

    def forward(self, x):
        x = self.matmul(x)
        if self.training:
            x = self.dropout(x)
            x = torch.softmax(x, dim=1)
        else:
            # Use optimized softmax kernel in eval mode
            x = self.fused_module.softmax_hip(x)
        return x
