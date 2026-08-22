import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

softmax_cpp_source = """
#include <hip/hip_runtime.h>

#define WARP_SIZE 64

__device__ float warp_max(float val) {
    val = max(val, __shfl_down(val, 32));
    val = max(val, __shfl_down(val, 16));
    val = max(val, __shfl_down(val, 8));
    val = max(val, __shfl_down(val, 4));
    val = max(val, __shfl_down(val, 2));
    val = max(val, __shfl_down(val, 1));
    return val;
}

__device__ float warp_sum(float val) {
    val += __shfl_down(val, 32);
    val += __shfl_down(val, 16);
    val += __shfl_down(val, 8);
    val += __shfl_down(val, 4);
    val += __shfl_down(val, 2);
    val += __shfl_down(val, 1);
    return val;
}

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ out, int batch_size, int dim) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row >= batch_size) return;
    
    const float* row_ptr = x + row * dim;
    float* out_ptr = out + row * dim;
    
    // Load chunks of data per thread - each thread processes multiple elements
    int chunk_size = (dim + blockDim.x - 1) / blockDim.x;
    int start = tid * chunk_size;
    int end = min(start + chunk_size, dim);
    
    // Compute max per thread first
    float max_val = -1e20f;
    for (int i = start; i < end; i++) {
        max_val = max(max_val, row_ptr[i]);
    }
    
    // Warp reduction for max
    max_val = warp_max(max_val);
    
    // Only need block reduce if more than one warp
    if (blockDim.x > WARP_SIZE) {
        extern __shared__ float smem[];
        int lane_id = tid % WARP_SIZE;
        int warp_id = tid / WARP_SIZE;
        
        if (lane_id == 0) smem[warp_id] = max_val;
        __syncthreads();
        
        if (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
            max_val = smem[tid];
        }
        max_val = warp_max(max_val);
        if (tid == 0) smem[0] = max_val;
        __syncthreads();
        max_val = smem[0];
    }
    
    // Compute sum of exp(x - max) per thread
    float sum_val = 0.0f;
    for (int i = start; i < end; i++) {
        sum_val += expf(row_ptr[i] - max_val);
    }
    
    // Warp reduction for sum
    sum_val = warp_sum(sum_val);
    
    if (blockDim.x > WARP_SIZE) {
        extern __shared__ float smem[];
        int lane_id = tid % WARP_SIZE;
        int warp_id = tid / WARP_SIZE;
        
        if (lane_id == 0) smem[warp_id] = sum_val;
        __syncthreads();
        
        if (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
            sum_val = smem[tid];
        }
        sum_val = warp_sum(sum_val);
        if (tid == 0) smem[0] = sum_val;
        __syncthreads();
        sum_val = smem[0];
    }
    
    float inv_sum = 1.0f / sum_val;
    
    // Compute final softmax
    for (int i = start; i < end; i++) {
        out_ptr[i] = expf(row_ptr[i] - max_val) * inv_sum;
    }
}

torch::Tensor softmax_hip(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto dim = x.size(1);
    
    auto out = torch::zeros_like(x);
    
    const int BLOCK_SIZE = 256;  // Works well for GPU
    dim3 block(BLOCK_SIZE);
    dim3 grid(batch_size);
    
    // Shared memory for block-level reduction (one value per warp)
    int smem_size = ((BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE) * sizeof(float);
    
    hipLaunchKernelGGL(HIP_KERNEL_NAME(softmax_kernel), 
                       grid, block, smem_size, 0,
                       x.data_ptr<float>(), out.data_ptr<float>(), 
                       batch_size, dim);
    
    return out;
}
"""

softmax_module = load_inline(
    name="softmax_hip",
    cpp_sources=softmax_cpp_source,
    functions=["softmax_hip"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax_hip = softmax_module
        
    def forward(self, x):
        return self.softmax_hip.softmax_hip(x)