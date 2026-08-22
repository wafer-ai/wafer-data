import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused dropout + softmax kernel
dropout_softmax_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <curand_kernel.h>

// Fused dropout + softmax kernel using online softmax algorithm
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
    const int num_threads = blockDim.x;
    
    const float* row_input = input + row * num_features;
    float* row_output = output + row * num_features;
    
    // Initialize random state for this thread
    curandStatePhilox4_32_10_t state;
    if (training) {
        curand_init(seed, row * num_features + tid, 0, &state);
    }
    
    // Online softmax: first pass - find max and compute sum
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    
    // Each thread handles multiple elements
    for (int i = tid; i < num_features; i += num_threads) {
        float val = row_input[i];
        
        // Apply dropout during training
        if (training) {
            float rand_val = curand_uniform(&state);
            if (rand_val < dropout_p) {
                val = 0.0f;
            } else {
                val = val * scale;
            }
        }
        
        // Online softmax update
        if (val > max_val) {
            sum_exp = sum_exp * expf(max_val - val) + 1.0f;
            max_val = val;
        } else {
            sum_exp += expf(val - max_val);
        }
    }
    
    // Shared memory for reduction
    extern __shared__ float shared[];
    float* shared_max = shared;
    float* shared_sum = shared + num_threads;
    
    shared_max[tid] = max_val;
    shared_sum[tid] = sum_exp;
    __syncthreads();
    
    // Reduce max across threads
    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_max = shared_max[tid + stride];
            float other_sum = shared_sum[tid + stride];
            float my_max = shared_max[tid];
            float my_sum = shared_sum[tid];
            
            if (other_max > my_max) {
                shared_sum[tid] = my_sum * expf(my_max - other_max) + other_sum;
                shared_max[tid] = other_max;
            } else {
                shared_sum[tid] = my_sum + other_sum * expf(other_max - my_max);
            }
        }
        __syncthreads();
    }
    
    float global_max = shared_max[0];
    float global_sum = shared_sum[0];
    
    // Re-initialize random state for second pass
    if (training) {
        curand_init(seed, row * num_features + tid, 0, &state);
    }
    
    // Second pass: compute softmax output
    for (int i = tid; i < num_features; i += num_threads) {
        float val = row_input[i];
        
        // Apply dropout (same random sequence)
        if (training) {
            float rand_val = curand_uniform(&state);
            if (rand_val < dropout_p) {
                val = 0.0f;
            } else {
                val = val * scale;
            }
        }
        
        row_output[i] = expf(val - global_max) / global_sum;
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
    
    const int threads = 256;
    const int shared_mem = 2 * threads * sizeof(float);
    
    // Generate random seed
    unsigned long long seed = training ? std::rand() : 0;
    float scale = training ? 1.0f / (1.0f - dropout_p) : 1.0f;
    
    dropout_softmax_forward_kernel<<<batch_size, threads, shared_mem>>>(
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
