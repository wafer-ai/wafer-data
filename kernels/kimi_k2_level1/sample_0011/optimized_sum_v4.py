import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Simpler and more reliable HIP kernel for sum reduction
sum_reduction_hip_code = """
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define BLOCK_SIZE 256

// Simple and reliable sum reduction kernel
__global__ void sum_reduction_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int reduce_dim,
    int dim2
) {
    int batch_idx = blockIdx.x;
    int dim2_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || dim2_idx >= dim2) return;
    
    __shared__ float shared_mem[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    
    // Calculate starting index for this thread's work
    // Input layout: [batch, reduce_dim, dim2]
    int base_idx = batch_idx * reduce_dim * dim2 + dim2_idx;
    
    float sum = 0.0f;
    
    // Each thread processes elements in the reduction dimension
    // with grid-stride loop for better memory coalescing
    for (int i = tid; i < reduce_dim; i += BLOCK_SIZE) {
        int idx = base_idx + i * dim2;
        sum += input[idx];
    }
    
    // Store in shared memory
    shared_mem[tid] = sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    // Using binary reduction pattern
    #pragma unroll
    for (int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    // Last warp: no sync needed (assuming all threads in warp execute in lockstep)
    if (tid < 32) {
        #pragma unroll
        for (int s = 32; s > 0; s >>= 1) {
            shared_mem[tid] += shared_mem[tid + s];
        }
    }
    
    // Thread 0 writes the final result
    if (tid == 0) {
        int out_idx = batch_idx * dim2 + dim2_idx;
        output[out_idx] = shared_mem[0];
    }
}

torch::Tensor sum_reduction_hip(torch::Tensor input, int dim) {
    auto sizes = input.sizes();
    int batch_size = sizes[0];
    int reduce_dim = sizes[1];
    int dim2 = sizes[2];
    
    // Allocate output with shape [batch_size, 1, dim2]
    auto output_sizes = input.sizes().vec();
    output_sizes[dim] = 1;
    auto output = torch::zeros(output_sizes, input.options());
    
    dim3 grid(batch_size, dim2);
    dim3 block(BLOCK_SIZE);
    
    hipLaunchKernelGGL(
        sum_reduction_kernel,
        grid,
        block,
        0, 0,
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        reduce_dim,
        dim2
    );
    
    return output;
}
"""

sum_reduction_hip = load_inline(
    name="sum_reduction",
    cpp_sources=sum_reduction_hip_code,
    functions=["sum_reduction_hip"],
    verbose=True,
)

batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1

class ModelNew(nn.Module):
    """
    Optimized model that performs sum reduction over a specified dimension using custom HIP kernel.
    """
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.sum_reduction_hip = sum_reduction_hip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies sum reduction over the specified dimension.
        """
        if self.dim == 1 and x.dim() == 3:
            return self.sum_reduction_hip.sum_reduction_hip(x, self.dim)
        else:
            return torch.sum(x, dim=self.dim, keepdim=True)

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [reduce_dim]