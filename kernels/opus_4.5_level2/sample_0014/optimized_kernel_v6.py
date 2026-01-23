import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Two-pass GELU + Softmax kernel storing GELU values to avoid recomputation
fused_gelu_softmax_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define WARP_SIZE 64
#define BLOCK_SIZE 1024

// GELU approximation
__device__ __forceinline__ float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Two-pass approach: store GELU values first, then softmax
__global__ void fused_gelu_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float shared_vals[NUM_WARPS];
    __shared__ float global_val;
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    if (row >= rows) return;
    
    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;
    
    // Vector types for coalesced access
    int cols_vec = cols / 4;
    const float4* in_vec = reinterpret_cast<const float4*>(row_input);
    float4* out_vec = reinterpret_cast<float4*>(row_output);
    
    // Pass 1: Apply GELU, store result, find max
    float local_max = -INFINITY;
    
    for (int i = tid; i < cols_vec; i += BLOCK_SIZE) {
        float4 v = in_vec[i];
        float4 g;
        g.x = gelu(v.x);
        g.y = gelu(v.y);
        g.z = gelu(v.z);
        g.w = gelu(v.w);
        out_vec[i] = g;  // Store GELU results
        local_max = fmaxf(local_max, fmaxf(fmaxf(g.x, g.y), fmaxf(g.z, g.w)));
    }
    
    // Warp reduction for max
    local_max = warp_reduce_max(local_max);
    if (lane_id == 0) shared_vals[warp_id] = local_max;
    __syncthreads();
    
    if (tid < NUM_WARPS) local_max = shared_vals[tid];
    else local_max = -INFINITY;
    
    if (warp_id == 0) {
        local_max = warp_reduce_max(local_max);
        if (lane_id == 0) global_val = local_max;
    }
    __syncthreads();
    
    float max_val = global_val;
    
    // Pass 2: exp(x - max) and sum
    float local_sum = 0.0f;
    
    for (int i = tid; i < cols_vec; i += BLOCK_SIZE) {
        float4 g = out_vec[i];  // Read stored GELU values
        float4 e;
        e.x = expf(g.x - max_val);
        e.y = expf(g.y - max_val);
        e.z = expf(g.z - max_val);
        e.w = expf(g.w - max_val);
        out_vec[i] = e;
        local_sum += e.x + e.y + e.z + e.w;
    }
    
    // Warp reduction for sum
    local_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) shared_vals[warp_id] = local_sum;
    __syncthreads();
    
    if (tid < NUM_WARPS) local_sum = shared_vals[tid];
    else local_sum = 0.0f;
    
    if (warp_id == 0) {
        local_sum = warp_reduce_sum(local_sum);
        if (lane_id == 0) global_val = local_sum;
    }
    __syncthreads();
    
    float inv_sum = 1.0f / global_val;
    
    // Pass 3: Normalize
    for (int i = tid; i < cols_vec; i += BLOCK_SIZE) {
        float4 e = out_vec[i];
        e.x *= inv_sum;
        e.y *= inv_sum;
        e.z *= inv_sum;
        e.w *= inv_sum;
        out_vec[i] = e;
    }
}

torch::Tensor fused_gelu_softmax_hip(torch::Tensor input) {
    auto rows = input.size(0);
    auto cols = input.size(1);
    auto output = torch::empty_like(input);
    
    dim3 grid(rows);
    dim3 block(BLOCK_SIZE);
    
    fused_gelu_softmax_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols
    );
    
    return output;
}
"""

fused_gelu_softmax_cpp = """
torch::Tensor fused_gelu_softmax_hip(torch::Tensor input);
"""

fused_gelu_softmax = load_inline(
    name="fused_gelu_softmax",
    cpp_sources=fused_gelu_softmax_cpp,
    cuda_sources=fused_gelu_softmax_source,
    functions=["fused_gelu_softmax_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        x = fused_gelu_softmax.fused_gelu_softmax_hip(x)
        return x


def get_inputs():
    return [torch.rand(1024, 8192).cuda()]


def get_init_inputs():
    return [8192, 8192]
