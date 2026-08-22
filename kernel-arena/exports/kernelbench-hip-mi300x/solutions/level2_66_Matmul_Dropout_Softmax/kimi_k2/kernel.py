import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

# Simple but efficient HIP kernel for softmax
hip_code = """
#include <hip/hip_runtime.h>

// Optimized softmax kernel
// Each block processes one row (batch element)
__global__ void softmax_kernel(
    float* input_output,
    int batch_size,
    int features
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;
    
    __shared__ float smem[32];
    
    // Step 1: Find maximum value in the row using warp-level reduction
    float local_max = -INFINITY;
    for (int i = tid; i < features; i += blockDim.x) {
        float val = input_output[row * features + i];
        if (val > local_max) {
            local_max = val;
        }
    }
    
    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_max = __shfl_down(local_max, offset);
        if (other_max > local_max) {
            local_max = other_max;
        }
    }
    
    // First thread of each warp writes to shared memory
    if (lane_id == 0) {
        smem[warp_id] = local_max;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (warp_id == 0) {
        local_max = (lane_id < num_warps) ? smem[lane_id] : -INFINITY;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_max = __shfl_down(local_max, offset);
            if (other_max > local_max) {
                local_max = other_max;
            }
        }
        
        // Store final max in shared memory
        if (lane_id == 0) {
            smem[0] = local_max;
        }
    }
    __syncthreads();
    
    float row_max = smem[0];
    
    // Step 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < features; i += blockDim.x) {
        float exp_val = expf(input_output[row * features + i] - row_max);
        input_output[row * features + i] = exp_val;
        local_sum += exp_val;
    }
    
    // Warp-level reduction for sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down(local_max, offset);
    }
    
    // First thread of each warp writes to shared memory
    if (lane_id == 0) {
        smem[warp_id] = local_sum;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (warp_id == 0) {
        local_sum = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_down(local_max, offset);
        }
        
        // Store final sum in shared memory
        if (lane_id == 0) {
            smem[0] = local_sum;
        }
    }
    __syncthreads();
    
    float row_sum = smem[0];
    
    // Step 3: Normalize
    for (int i = tid; i < features; i += blockDim.x) {
        input_output[row * features + i] /= row_sum;
    }
}

// Wrapper functions
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor softmax_hip(torch::Tensor input) {
    CHECK_INPUT(input);
    
    auto batch_size = input.size(0);
    auto features = input.size(1);
    
    // Best configuration for MI300X
    int threads = 256;  // 8 warps per block
    int blocks = batch_size;
    
    auto output = input.clone();
    
    hipLaunchKernelGGL(
        softmax_kernel,
        blocks,
        threads,
        0,  // No dynamic shared memory needed
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
    functions=["softmax_hip"],
    verbose=True,
    extra_cflags=["-O3", "-D__HIP_PLATFORM_AMD__"]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p
        
        # Use PyTorch's highly optimized layers
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)
        self.custom_ops = custom_ops
        
        # Enable TF32 for better performance (if available)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    def forward(self, x):
        # Use PyTorch's optimized linear layer
        x = self.linear(x)
        
        # Use PyTorch's optimized dropout
        x = self.dropout(x)
        
        # Use custom optimized softmax
        x = self.custom_ops.softmax_hip(x.contiguous())
        
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