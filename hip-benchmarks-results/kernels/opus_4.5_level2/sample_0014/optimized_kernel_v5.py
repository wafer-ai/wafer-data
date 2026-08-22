import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized Fused GELU + Softmax kernel using shared memory
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

// Fused GELU + Softmax kernel using tiled approach
// Process each row in tiles to utilize shared memory
__global__ void fused_gelu_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float shared_max[NUM_WARPS];
    __shared__ float shared_sum[NUM_WARPS];
    __shared__ float row_max;
    __shared__ float row_sum;
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    if (row >= rows) return;
    
    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;
    
    // Step 1: Find max with vectorized loads
    float local_max = -INFINITY;
    
    int cols_vec = cols / 4;
    const float4* row_input_vec = reinterpret_cast<const float4*>(row_input);
    
    for (int i = tid; i < cols_vec; i += BLOCK_SIZE) {
        float4 v = row_input_vec[i];
        local_max = fmaxf(local_max, gelu(v.x));
        local_max = fmaxf(local_max, gelu(v.y));
        local_max = fmaxf(local_max, gelu(v.z));
        local_max = fmaxf(local_max, gelu(v.w));
    }
    
    // Warp reduction for max
    local_max = warp_reduce_max(local_max);
    
    if (lane_id == 0) {
        shared_max[warp_id] = local_max;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (tid < NUM_WARPS) {
        local_max = shared_max[tid];
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
    
    // Step 2: Compute exp(gelu(x) - max), store, and sum
    float local_sum = 0.0f;
    float4* row_output_vec = reinterpret_cast<float4*>(row_output);
    
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
    
    // Warp reduction for sum
    local_sum = warp_reduce_sum(local_sum);
    
    if (lane_id == 0) {
        shared_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    if (tid < NUM_WARPS) {
        local_sum = shared_sum[tid];
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
    
    float inv_sum = 1.0f / row_sum;
    
    // Step 3: Normalize
    for (int i = tid; i < cols_vec; i += BLOCK_SIZE) {
        float4 v = row_output_vec[i];
        v.x *= inv_sum;
        v.y *= inv_sum;
        v.z *= inv_sum;
        v.w *= inv_sum;
        row_output_vec[i] = v;
    }
}

torch::Tensor fused_gelu_softmax_hip(torch::Tensor input) {
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
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
