import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

# Optimized HIP kernel - focus on efficient softmax without expensive dropout fusion
hip_code = """
#include <hip/hip_runtime.h>

// Optimized softmax kernel using warp-level reduction and vectorized loads
// Each warp processes a set of features, all warps in block cooperate on one row
__global__ void softmax_kernel_optimized(
    float* input_output,
    int batch_size,
    int features
) {
    int row = blockIdx.x;
    int laneId = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;
    int warpsPerBlock = blockDim.x / 32;
    
    // Use shared memory for inter-warp communication
    __shared__ float smem[32];
    __shared__ float row_max;
    __shared__ float row_sum;
    
    if (threadIdx.x == 0) {
        row_max = -INFINITY;
        row_sum = 0.0f;
    }
    __syncthreads();
    
    // Step 1: Find row maximum using warp-level reduction
    float local_max = -INFINITY;
    
    // Each thread processes multiple elements with stride
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float val = input_output[row * features + i];
        local_max = fmaxf(local_max, val);
    }
    
    // Warp-level reduction for max
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down(local_max, offset));
    }
    
    // First thread in each warp writes warp's max to shared memory
    if (laneId == 0) {
        atomicMaxFloat(&row_max, local_max);
    }
    __syncthreads();
    
    // Step 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    float max_val = row_max;
    
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float exp_val = expf(input_output[row * features + i] - max_val);
        input_output[row * features + i] = exp_val;
        local_sum += exp_val;
    }
    
    // Warp-level reduction for sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down(local_sum, offset);
    }
    
    // First thread in each warp adds to global sum
    if (laneId == 0) {
        atomicAdd(&row_sum, local_sum);
    }
    __syncthreads();
    
    // Step 3: Normalize
    float sum_val = row_sum;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        input_output[row * features + i] /= sum_val;
    }
}

// Fallback atomicMax for floats (not natively supported)
__device__ void atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

// Wrapper functions
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor optimized_softmax_hip(torch::Tensor input) {
    CHECK_INPUT(input);
    
    auto batch_size = input.size(0);
    auto features = input.size(1);
    
    // Use 256 threads per block for MI300X (optimal)
    int threads = 256;
    int blocks = batch_size;
    
    auto output = input.clone();
    
    hipLaunchKernelGGL(
        softmax_kernel_optimized,
        blocks,
        threads,
        0,  // No shared memory needed beyond what we declared
        0,  // Default stream
        output.data_ptr<float>(),
        batch_size,
        features
    );
    
    return output;
}
"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=hip_code,
    functions=["optimized_softmax_hip"],
    verbose=True,
    extra_cflags=["-O3", "-D__HIP_PLATFORM_AMD__"],
    extra_ldflags=["-lrocrand"]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p
        
        # Use PyTorch's linear layer (highly optimized)
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)
        self.custom_ops = custom_ops
        
    def forward(self, x):
        # Use PyTorch's optimized linear layer
        x = self.linear(x)
        
        # Use PyTorch's optimized dropout
        x = self.dropout(x)
        
        # Use custom optimized softmax
        x = self.custom_ops.optimized_softmax_hip(x.contiguous())
        
        return x

# Test inputs
batch_size = 128
in_features = 16384
out_features = 16384
dropout_p = 0.2

def get_inputs():
    return [torch.rand(batch_size, in_features, device='cuda')]

def get_init_inputs():
    return [in_features, out_features, dropout_p]