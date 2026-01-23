import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused dropout + softmax kernel
fused_dropout_softmax_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include <curand_kernel.h>

// Fused dropout + softmax kernel
// Each block handles one row of the input
__global__ void fused_dropout_softmax_kernel(
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
    const int block_size = blockDim.x;
    
    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;
    
    // Initialize random state for this thread
    curandStatePhilox4_32_10_t state;
    if (training) {
        curand_init(seed, row * cols + tid, 0, &state);
    }
    
    extern __shared__ float shared_mem[];
    float* shared_max = shared_mem;  // blockDim.x floats
    float* shared_sum = shared_mem + block_size;  // blockDim.x floats
    
    // Pass 1: Find max value (with dropout applied) for numerical stability
    float local_max = -INFINITY;
    for (int i = tid; i < cols; i += block_size) {
        float val = row_input[i];
        if (training) {
            // Apply dropout - we need deterministic behavior, so we need to regenerate
            curandStatePhilox4_32_10_t temp_state;
            curand_init(seed, row * cols + i, 0, &temp_state);
            float rand_val = curand_uniform(&temp_state);
            if (rand_val < dropout_p) {
                val = -INFINITY;  // This will be effectively 0 after softmax
            } else {
                val = val * scale;
            }
        }
        if (val > local_max) local_max = val;
    }
    
    shared_max[tid] = local_max;
    __syncthreads();
    
    // Reduce to find global max
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_max[tid + s] > shared_max[tid]) {
                shared_max[tid] = shared_max[tid + s];
            }
        }
        __syncthreads();
    }
    float row_max = shared_max[0];
    __syncthreads();
    
    // Pass 2: Compute sum of exp(x - max)
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += block_size) {
        float val = row_input[i];
        if (training) {
            curandStatePhilox4_32_10_t temp_state;
            curand_init(seed, row * cols + i, 0, &temp_state);
            float rand_val = curand_uniform(&temp_state);
            if (rand_val < dropout_p) {
                val = -INFINITY;
            } else {
                val = val * scale;
            }
        }
        local_sum += expf(val - row_max);
    }
    
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    // Reduce to find global sum
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    float row_sum = shared_sum[0];
    __syncthreads();
    
    // Pass 3: Compute softmax output
    for (int i = tid; i < cols; i += block_size) {
        float val = row_input[i];
        if (training) {
            curandStatePhilox4_32_10_t temp_state;
            curand_init(seed, row * cols + i, 0, &temp_state);
            float rand_val = curand_uniform(&temp_state);
            if (rand_val < dropout_p) {
                val = -INFINITY;
            } else {
                val = val * scale;
            }
        }
        row_output[i] = expf(val - row_max) / row_sum;
    }
}

torch::Tensor fused_dropout_softmax_hip(torch::Tensor input, float dropout_p, bool training) {
    auto output = torch::empty_like(input);
    
    const int rows = input.size(0);
    const int cols = input.size(1);
    
    const int block_size = 256;
    const int shared_mem_size = 2 * block_size * sizeof(float);
    
    float scale = 1.0f / (1.0f - dropout_p);
    
    // Generate random seed
    unsigned long long seed = training ? (unsigned long long)(rand()) : 0;
    
    hipLaunchKernelGGL(fused_dropout_softmax_kernel, 
        dim3(rows), dim3(block_size), shared_mem_size, 0,
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
    name="fused_dropout_softmax",
    cpp_sources=fused_dropout_softmax_cpp,
    cuda_sources=fused_dropout_softmax_source,
    functions=["fused_dropout_softmax_hip"],
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
        self.fused_module = fused_module

    def forward(self, x):
        x = self.matmul(x)
        x = self.fused_module.fused_dropout_softmax_hip(x, self.dropout_p, self.training)
        return x
