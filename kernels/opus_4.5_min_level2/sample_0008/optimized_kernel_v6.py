import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused dropout + softmax kernel
dropout_softmax_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <curand_kernel.h>

#define WARP_SIZE 64
#define BLOCK_SIZE 256

// Warp-level reduction for max
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

// Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Fused dropout + softmax kernel - writes dropout values to output first, then softmax in-place
__global__ void dropout_softmax_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int num_features,
    const float dropout_p,
    const float scale,
    const unsigned long long seed,
    const bool training
) {
    const int row = blockIdx.x;
    if (row >= batch_size) return;
    
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int num_warps = BLOCK_SIZE / WARP_SIZE;
    
    const float* row_input = input + row * num_features;
    float* row_output = output + row * num_features;
    
    __shared__ float warp_max[BLOCK_SIZE / WARP_SIZE];
    __shared__ float warp_sum[BLOCK_SIZE / WARP_SIZE];
    __shared__ float shared_max;
    __shared__ float shared_sum;
    
    // Initialize random state for this thread
    curandStatePhilox4_32_10_t state;
    if (training) {
        curand_init(seed, row * num_features + tid, 0, &state);
    }
    
    // First pass: apply dropout and write to output, compute max
    float thread_max = -INFINITY;
    
    for (int i = tid; i < num_features; i += BLOCK_SIZE) {
        float val = row_input[i];
        
        if (training) {
            float rand_val = curand_uniform(&state);
            if (rand_val < dropout_p) {
                val = 0.0f;
            } else {
                val = val * scale;
            }
        }
        // Store dropout result
        row_output[i] = val;
        thread_max = fmaxf(thread_max, val);
    }
    
    // Warp reduction for max
    float warp_max_val = warp_reduce_max(thread_max);
    if (lane_id == 0) {
        warp_max[warp_id] = warp_max_val;
    }
    __syncthreads();
    
    // Final reduction across warps (done by first warp)
    if (warp_id == 0) {
        float val = (tid < num_warps) ? warp_max[tid] : -INFINITY;
        val = warp_reduce_max(val);
        if (tid == 0) {
            shared_max = val;
        }
    }
    __syncthreads();
    
    float global_max = shared_max;
    
    // Second pass: compute sum of exp(x - max) - read from output
    float thread_sum = 0.0f;
    
    for (int i = tid; i < num_features; i += BLOCK_SIZE) {
        float val = row_output[i];
        thread_sum += expf(val - global_max);
    }
    
    // Warp reduction for sum
    float warp_sum_val = warp_reduce_sum(thread_sum);
    if (lane_id == 0) {
        warp_sum[warp_id] = warp_sum_val;
    }
    __syncthreads();
    
    // Final reduction across warps (done by first warp)
    if (warp_id == 0) {
        float val = (tid < num_warps) ? warp_sum[tid] : 0.0f;
        val = warp_reduce_sum(val);
        if (tid == 0) {
            shared_sum = val;
        }
    }
    __syncthreads();
    
    float inv_sum = 1.0f / shared_sum;
    
    // Third pass: compute final softmax output
    for (int i = tid; i < num_features; i += BLOCK_SIZE) {
        float val = row_output[i];
        row_output[i] = expf(val - global_max) * inv_sum;
    }
}

torch::Tensor dropout_softmax_forward(
    torch::Tensor input,
    float dropout_p,
    bool training
) {
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    
    auto output = torch::empty_like(input);
    
    // Generate random seed
    unsigned long long seed = training ? (unsigned long long)std::rand() : 0;
    float scale = training ? 1.0f / (1.0f - dropout_p) : 1.0f;
    
    dropout_softmax_forward_kernel<<<batch_size, BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_features,
        dropout_p,
        scale,
        seed,
        training
    );
    
    return output;
}
"""

dropout_softmax_cpp = """
torch::Tensor dropout_softmax_forward(torch::Tensor input, float dropout_p, bool training);
"""

dropout_softmax = load_inline(
    name="dropout_softmax",
    cpp_sources=dropout_softmax_cpp,
    cuda_sources=dropout_softmax_source,
    functions=["dropout_softmax_forward"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    """
    Optimized model with fused dropout + softmax kernel.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout_p = dropout_p
        self.dropout_softmax = dropout_softmax

    def forward(self, x):
        x = self.matmul(x)
        x = self.dropout_softmax.dropout_softmax_forward(x, self.dropout_p, self.training)
        return x
