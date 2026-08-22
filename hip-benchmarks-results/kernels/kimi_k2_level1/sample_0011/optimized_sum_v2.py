import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized HIP kernel for sum reduction over dimension 1
sum_reduction_hip_code = """
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define WARP_SIZE 64  // MI300X has 64 threads per wavefront
#define BLOCK_SIZE 256  // Optimal block size
#define VEC_SIZE 4  // Vectorized loads of 4 floats

// Optimized kernel for sum reduction over dimension 1 with vectorized loads
// Input shape: [batch_size, reduce_dim, dim2]
// Output shape: [batch_size, 1, dim2]
__global__ void sum_reduction_optimized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int reduce_dim,
    int dim2
) {
    // 2D grid: x = batch_idx, y = dim2_idx
    int batch_idx = blockIdx.x;
    int dim2_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || dim2_idx >= dim2) return;
    
    __shared__ float shared_mem[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = BLOCK_SIZE / WARP_SIZE;
    
    // Calculate starting pointer for this thread's data
    const float* input_ptr = input + batch_idx * (reduce_dim * dim2) + dim2_idx;
    
    float sum = 0.0f;
    
    // Vectorized loads for better memory bandwidth utilization
    // Each thread processes multiple elements with vectorized loads
    int i = tid * VEC_SIZE;
    
    // Main loop with vectorized loads
    for (; i + VEC_SIZE - 1 < reduce_dim; i += BLOCK_SIZE * VEC_SIZE) {
        float4 vec_data = *reinterpret_cast<const float4*>(input_ptr + i * dim2);
        sum += vec_data.x + vec_data.y + vec_data.z + vec_data.w;
    }
    
    // Handle remaining elements
    for (; i < reduce_dim; i++) {
        sum += input_ptr[i * dim2];
    }
    
    shared_mem[tid] = sum;
    __syncthreads();
    
    // Warp-level reduction using shuffle operations (more efficient than shared memory)
    // First, reduce within each warp
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    
    // Write warp sum to shared memory
    if (lane_id == 0) {
        shared_mem[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction using the first warp
    if (warp_id == 0) {
        sum = (lane_id < num_warps) ? shared_mem[lane_id] : 0.0f;
        
        #pragma unroll
        for (int offset = num_warps / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down(sum, offset);
        }
        
        // Thread 0 writes the final result
        if (lane_id == 0) {
            int out_idx = batch_idx * dim2 + dim2_idx;
            output[out_idx] = sum;
        }
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
    
    // Optimal launch configuration for MI300X
    // Use 2D grid for better occupancy and L2 cache utilization
    dim3 grid(batch_size, dim2);
    dim3 block(BLOCK_SIZE);
    
    hipLaunchKernelGGL(
        sum_reduction_optimized_kernel,
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
    name="sum_reduction_optimized",
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