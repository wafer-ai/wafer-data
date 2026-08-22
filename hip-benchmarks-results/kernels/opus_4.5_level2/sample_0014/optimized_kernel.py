import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused GELU + Softmax kernel
fused_gelu_softmax_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define WARP_SIZE 64
#define BLOCK_SIZE 256

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
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

// Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Fused GELU + Softmax kernel - each block handles one row
__global__ void fused_gelu_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    __shared__ float warp_max[BLOCK_SIZE / WARP_SIZE];
    __shared__ float warp_sum[BLOCK_SIZE / WARP_SIZE];
    __shared__ float row_max;
    __shared__ float row_sum;
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = BLOCK_SIZE / WARP_SIZE;
    
    if (row >= rows) return;
    
    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;
    
    // Step 1: Apply GELU and find max for numerical stability
    float local_max = -INFINITY;
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float val = gelu(row_input[i]);
        local_max = fmaxf(local_max, val);
    }
    
    // Warp-level reduction for max
    local_max = warp_reduce_max(local_max);
    
    // Store warp results
    if (lane_id == 0) {
        warp_max[warp_id] = local_max;
    }
    __syncthreads();
    
    // Final reduction for max (first warp)
    if (tid < num_warps) {
        local_max = warp_max[tid];
    } else {
        local_max = -INFINITY;
    }
    if (warp_id == 0) {
        local_max = warp_reduce_max(local_max);
        if (lane_id == 0) {
            row_max = local_max;
        }
    }
    __syncthreads();
    
    float max_val = row_max;
    
    // Step 2: Compute exp(gelu(x) - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float val = gelu(row_input[i]);
        float exp_val = expf(val - max_val);
        row_output[i] = exp_val;  // Temporarily store exp values
        local_sum += exp_val;
    }
    
    // Warp-level reduction for sum
    local_sum = warp_reduce_sum(local_sum);
    
    if (lane_id == 0) {
        warp_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    // Final reduction for sum
    if (tid < num_warps) {
        local_sum = warp_sum[tid];
    } else {
        local_sum = 0.0f;
    }
    if (warp_id == 0) {
        local_sum = warp_reduce_sum(local_sum);
        if (lane_id == 0) {
            row_sum = local_sum;
        }
    }
    __syncthreads();
    
    float sum_val = row_sum;
    float inv_sum = 1.0f / sum_val;
    
    // Step 3: Normalize
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
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
