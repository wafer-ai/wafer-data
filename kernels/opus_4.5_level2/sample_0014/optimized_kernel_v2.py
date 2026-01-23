import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused GELU + Softmax kernel with vectorized loads
fused_gelu_softmax_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define WARP_SIZE 64
#define BLOCK_SIZE 512

// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ __forceinline__ float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Warp-level reduction for max
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

// Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Block-level reduction for max
__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float warp_results[BLOCK_SIZE / WARP_SIZE];
    
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int num_warps = BLOCK_SIZE / WARP_SIZE;
    
    val = warp_reduce_max(val);
    
    if (lane_id == 0) {
        warp_results[warp_id] = val;
    }
    __syncthreads();
    
    val = (threadIdx.x < num_warps) ? warp_results[threadIdx.x] : -INFINITY;
    
    if (warp_id == 0) {
        val = warp_reduce_max(val);
    }
    
    __shared__ float block_max;
    if (threadIdx.x == 0) {
        block_max = val;
    }
    __syncthreads();
    
    return block_max;
}

// Block-level reduction for sum
__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float warp_results[BLOCK_SIZE / WARP_SIZE];
    
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int num_warps = BLOCK_SIZE / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    
    if (lane_id == 0) {
        warp_results[warp_id] = val;
    }
    __syncthreads();
    
    val = (threadIdx.x < num_warps) ? warp_results[threadIdx.x] : 0.0f;
    
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }
    
    __shared__ float block_sum;
    if (threadIdx.x == 0) {
        block_sum = val;
    }
    __syncthreads();
    
    return block_sum;
}

// Fused GELU + Softmax kernel - each block handles one row
// Processes 4 elements per thread per iteration (float4)
__global__ void fused_gelu_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row >= rows) return;
    
    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;
    
    int cols_vec = cols / 4;
    const float4* row_input_vec = reinterpret_cast<const float4*>(row_input);
    float4* row_output_vec = reinterpret_cast<float4*>(row_output);
    
    // Step 1: Apply GELU and find max for numerical stability
    float local_max = -INFINITY;
    for (int i = tid; i < cols_vec; i += BLOCK_SIZE) {
        float4 v = row_input_vec[i];
        float g0 = gelu(v.x);
        float g1 = gelu(v.y);
        float g2 = gelu(v.z);
        float g3 = gelu(v.w);
        local_max = fmaxf(local_max, fmaxf(fmaxf(g0, g1), fmaxf(g2, g3)));
    }
    // Handle remainder
    for (int i = cols_vec * 4 + tid; i < cols; i += BLOCK_SIZE) {
        float g = gelu(row_input[i]);
        local_max = fmaxf(local_max, g);
    }
    
    float max_val = block_reduce_max(local_max);
    
    // Step 2: Compute exp(gelu(x) - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < cols_vec; i += BLOCK_SIZE) {
        float4 v = row_input_vec[i];
        float4 out;
        out.x = expf(gelu(v.x) - max_val);
        out.y = expf(gelu(v.y) - max_val);
        out.z = expf(gelu(v.z) - max_val);
        out.w = expf(gelu(v.w) - max_val);
        row_output_vec[i] = out;
        local_sum += out.x + out.y + out.z + out.w;
    }
    // Handle remainder
    for (int i = cols_vec * 4 + tid; i < cols; i += BLOCK_SIZE) {
        float exp_val = expf(gelu(row_input[i]) - max_val);
        row_output[i] = exp_val;
        local_sum += exp_val;
    }
    
    float sum_val = block_reduce_sum(local_sum);
    float inv_sum = 1.0f / sum_val;
    
    // Step 3: Normalize
    for (int i = tid; i < cols_vec; i += BLOCK_SIZE) {
        float4 v = row_output_vec[i];
        v.x *= inv_sum;
        v.y *= inv_sum;
        v.z *= inv_sum;
        v.w *= inv_sum;
        row_output_vec[i] = v;
    }
    // Handle remainder
    for (int i = cols_vec * 4 + tid; i < cols; i += BLOCK_SIZE) {
        row_output[i] *= inv_sum;
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
