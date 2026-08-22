import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused dropout + softmax kernel with better memory access
fused_dropout_softmax_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hiprand/hiprand_kernel.h>
#include <cmath>

#define WARP_SIZE 64
#define BLOCK_SIZE 256

// Warp reduce max
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

// Warp reduce sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Online softmax kernel - one block per row
// Uses online algorithm to compute max and sum in a single pass
__global__ void fused_dropout_softmax_online_kernel(
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
    const int num_warps = BLOCK_SIZE / WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;
    
    __shared__ float shared_max[BLOCK_SIZE / WARP_SIZE];
    __shared__ float shared_sum[BLOCK_SIZE / WARP_SIZE];
    
    // Phase 1: Online max and sum computation
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    
    // Initialize random state if training
    hiprandStatePhilox4_32_10_t state;
    if (training) {
        hiprand_init(seed, (unsigned long long)(row * cols + tid), 0, &state);
    }
    
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float val = row_input[i];
        
        if (training) {
            float rand_val = hiprand_uniform(&state);
            if (rand_val < dropout_p) {
                val = -INFINITY;
            } else {
                val *= scale;
            }
        }
        
        // Online softmax update
        if (val > local_max) {
            local_sum = local_sum * expf(local_max - val) + 1.0f;
            local_max = val;
        } else {
            local_sum += expf(val - local_max);
        }
    }
    
    // Warp-level reduction for max
    float warp_max = warp_reduce_max(local_max);
    if (lane_id == 0) {
        shared_max[warp_id] = warp_max;
    }
    __syncthreads();
    
    // First warp reduces across warps
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared_max[lane_id] : -INFINITY;
        val = warp_reduce_max(val);
        if (lane_id == 0) {
            shared_max[0] = val;
        }
    }
    __syncthreads();
    float global_max = shared_max[0];
    
    // Adjust local_sum for global_max
    local_sum = local_sum * expf(local_max - global_max);
    
    // Warp-level reduction for sum
    float warp_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) {
        shared_sum[warp_id] = warp_sum;
    }
    __syncthreads();
    
    // First warp reduces across warps
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            shared_sum[0] = val;
        }
    }
    __syncthreads();
    float global_sum = shared_sum[0];
    float inv_sum = 1.0f / global_sum;
    
    // Phase 2: Compute final softmax values
    // Re-initialize random state for consistent dropout pattern
    if (training) {
        hiprand_init(seed, (unsigned long long)(row * cols + tid), 0, &state);
    }
    
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float val = row_input[i];
        
        if (training) {
            float rand_val = hiprand_uniform(&state);
            if (rand_val < dropout_p) {
                val = -INFINITY;
            } else {
                val *= scale;
            }
        }
        
        row_output[i] = expf(val - global_max) * inv_sum;
    }
}

torch::Tensor fused_dropout_softmax_hip(torch::Tensor input, float dropout_p, bool training) {
    auto output = torch::empty_like(input);
    
    const int rows = input.size(0);
    const int cols = input.size(1);
    
    float scale = 1.0f / (1.0f - dropout_p);
    
    // Generate random seed
    unsigned long long seed = training ? (unsigned long long)(rand()) : 0;
    
    hipLaunchKernelGGL(fused_dropout_softmax_online_kernel,
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
    name="fused_dropout_softmax_v2",
    cpp_sources=fused_dropout_softmax_cpp,
    cuda_sources=fused_dropout_softmax_source,
    functions=["fused_dropout_softmax_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    """
    Optimized model with fused dropout + softmax kernel using online softmax.
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
