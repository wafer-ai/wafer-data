import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused GELU + Softmax kernel with online softmax algorithm
fused_gelu_softmax_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define WARP_SIZE 64
#define BLOCK_SIZE 1024

// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ __forceinline__ float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Online softmax: combine max finding and sum computation
// This reduces passes through data
struct SoftmaxState {
    float max_val;
    float sum;
};

__device__ __forceinline__ SoftmaxState combine_states(SoftmaxState a, SoftmaxState b) {
    SoftmaxState result;
    result.max_val = fmaxf(a.max_val, b.max_val);
    result.sum = a.sum * expf(a.max_val - result.max_val) + 
                 b.sum * expf(b.max_val - result.max_val);
    return result;
}

__device__ __forceinline__ SoftmaxState warp_reduce_softmax(SoftmaxState state) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        SoftmaxState other;
        other.max_val = __shfl_xor(state.max_val, offset);
        other.sum = __shfl_xor(state.sum, offset);
        state = combine_states(state, other);
    }
    return state;
}

// Fused GELU + Softmax kernel using online softmax
// Single pass to compute max and sum together
__global__ void fused_gelu_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float shared_max[NUM_WARPS];
    __shared__ float shared_sum[NUM_WARPS];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    if (row >= rows) return;
    
    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;
    
    // Use float4 for coalesced memory access
    int cols_vec = cols / 4;
    const float4* row_input_vec = reinterpret_cast<const float4*>(row_input);
    
    // Online softmax: compute max and sum in single pass
    SoftmaxState local_state;
    local_state.max_val = -INFINITY;
    local_state.sum = 0.0f;
    
    for (int i = tid; i < cols_vec; i += BLOCK_SIZE) {
        float4 v = row_input_vec[i];
        float g0 = gelu(v.x);
        float g1 = gelu(v.y);
        float g2 = gelu(v.z);
        float g3 = gelu(v.w);
        
        // Online update for g0
        if (g0 > local_state.max_val) {
            local_state.sum = local_state.sum * expf(local_state.max_val - g0) + 1.0f;
            local_state.max_val = g0;
        } else {
            local_state.sum += expf(g0 - local_state.max_val);
        }
        
        // Online update for g1
        if (g1 > local_state.max_val) {
            local_state.sum = local_state.sum * expf(local_state.max_val - g1) + 1.0f;
            local_state.max_val = g1;
        } else {
            local_state.sum += expf(g1 - local_state.max_val);
        }
        
        // Online update for g2
        if (g2 > local_state.max_val) {
            local_state.sum = local_state.sum * expf(local_state.max_val - g2) + 1.0f;
            local_state.max_val = g2;
        } else {
            local_state.sum += expf(g2 - local_state.max_val);
        }
        
        // Online update for g3
        if (g3 > local_state.max_val) {
            local_state.sum = local_state.sum * expf(local_state.max_val - g3) + 1.0f;
            local_state.max_val = g3;
        } else {
            local_state.sum += expf(g3 - local_state.max_val);
        }
    }
    
    // Handle remainder
    for (int i = cols_vec * 4 + tid; i < cols; i += BLOCK_SIZE) {
        float g = gelu(row_input[i]);
        if (g > local_state.max_val) {
            local_state.sum = local_state.sum * expf(local_state.max_val - g) + 1.0f;
            local_state.max_val = g;
        } else {
            local_state.sum += expf(g - local_state.max_val);
        }
    }
    
    // Warp-level reduction
    local_state = warp_reduce_softmax(local_state);
    
    if (lane_id == 0) {
        shared_max[warp_id] = local_state.max_val;
        shared_sum[warp_id] = local_state.sum;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (tid < NUM_WARPS) {
        local_state.max_val = shared_max[tid];
        local_state.sum = shared_sum[tid];
    } else {
        local_state.max_val = -INFINITY;
        local_state.sum = 0.0f;
    }
    
    if (warp_id == 0) {
        local_state = warp_reduce_softmax(local_state);
    }
    
    __shared__ float final_max;
    __shared__ float final_sum;
    
    if (tid == 0) {
        final_max = local_state.max_val;
        final_sum = local_state.sum;
    }
    __syncthreads();
    
    float max_val = final_max;
    float inv_sum = 1.0f / final_sum;
    
    // Write output: compute gelu, then softmax
    float4* row_output_vec = reinterpret_cast<float4*>(row_output);
    
    for (int i = tid; i < cols_vec; i += BLOCK_SIZE) {
        float4 v = row_input_vec[i];
        float4 out;
        out.x = expf(gelu(v.x) - max_val) * inv_sum;
        out.y = expf(gelu(v.y) - max_val) * inv_sum;
        out.z = expf(gelu(v.z) - max_val) * inv_sum;
        out.w = expf(gelu(v.w) - max_val) * inv_sum;
        row_output_vec[i] = out;
    }
    
    // Handle remainder
    for (int i = cols_vec * 4 + tid; i < cols; i += BLOCK_SIZE) {
        row_output[i] = expf(gelu(row_input[i]) - max_val) * inv_sum;
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
