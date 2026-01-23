import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized softmax kernel using vectorized loads and online algorithm
softmax_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

#define WARP_SIZE 64
#define BLOCK_SIZE 256

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

// Softmax kernel optimized for large row sizes
// Uses float4 vectorization and warp-level reductions
__global__ void softmax_large_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int rows,
    const int cols
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    
    const int tid = threadIdx.x;
    const int num_warps = BLOCK_SIZE / WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;
    
    __shared__ float shared_max[BLOCK_SIZE / WARP_SIZE];
    __shared__ float shared_sum[BLOCK_SIZE / WARP_SIZE];
    
    // Vector processing
    const int cols_vec = cols / 4;
    const float4* input_vec = reinterpret_cast<const float4*>(row_input);
    float4* output_vec = reinterpret_cast<float4*>(row_output);
    
    // Phase 1: Find max using vectorized loads
    float local_max = -INFINITY;
    for (int i = tid; i < cols_vec; i += BLOCK_SIZE) {
        float4 val = input_vec[i];
        local_max = fmaxf(local_max, fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w)));
    }
    
    // Warp-level reduction for max
    float warp_max = warp_reduce_max(local_max);
    if (lane_id == 0) shared_max[warp_id] = warp_max;
    __syncthreads();
    
    // First warp reduces across warps
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared_max[lane_id] : -INFINITY;
        val = warp_reduce_max(val);
        if (lane_id == 0) shared_max[0] = val;
    }
    __syncthreads();
    float global_max = shared_max[0];
    
    // Phase 2: Sum of exp using vectorized loads
    float local_sum = 0.0f;
    for (int i = tid; i < cols_vec; i += BLOCK_SIZE) {
        float4 val = input_vec[i];
        local_sum += expf(val.x - global_max);
        local_sum += expf(val.y - global_max);
        local_sum += expf(val.z - global_max);
        local_sum += expf(val.w - global_max);
    }
    
    // Warp-level reduction for sum
    float warp_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) shared_sum[warp_id] = warp_sum;
    __syncthreads();
    
    // First warp reduces across warps
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) shared_sum[0] = val;
    }
    __syncthreads();
    float inv_sum = 1.0f / shared_sum[0];
    
    // Phase 3: Compute softmax using vectorized stores
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
    
    hipLaunchKernelGGL(softmax_large_kernel,
        dim3(rows), dim3(BLOCK_SIZE), 0, 0,
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols
    );
    
    return output;
}
"""

softmax_cpp = """
torch::Tensor softmax_hip(torch::Tensor input);
"""

softmax_module = load_inline(
    name="softmax_v5",
    cpp_sources=softmax_cpp,
    cuda_sources=softmax_source,
    functions=["softmax_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    """
    Optimized model with custom softmax kernel.
    Dropout is handled by PyTorch as it's harder to match exactly.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)
        self.softmax_module = softmax_module

    def forward(self, x):
        x = self.matmul(x)
        x = self.dropout(x)
        x = self.softmax_module.softmax_hip(x)
        return x
