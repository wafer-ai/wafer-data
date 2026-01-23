import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

softmax_hip_source = """
#include <hip/hip_runtime.h>

#define WARP_SIZE 64

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down(val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ out, int batch_size, int dim) {
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    const float* row_ptr = x + row * dim;
    float* out_ptr = out + row * dim;
    
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int warps_per_block = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    
    __shared__ float smem[16];  // Up to 8 warps
    
    // Phase 1: Find maximum with coalesced memory access
    float local_max = -INFINITY;
    int stride = blockDim.x;
    
    for (int i = tid; i < dim; i += stride) {
        local_max = fmaxf(local_max, row_ptr[i]);
    }
    
    // Warp-level reduction
    local_max = warp_reduce_max(local_max);
    
    // Cross-warp reduction
    if (lane_id == 0 && warp_id < 16) {
        smem[warp_id] = local_max;
    }
    __syncthreads();
    
    float max_val;
    if (tid < 16) {
        max_val = smem[tid];
        max_val = warp_reduce_max(max_val);
        if (tid == 0) smem[0] = max_val;
    }
    __syncthreads();
    max_val = smem[0];
    
    // Phase 2: Compute exp(x - max) and sum, with coalesced writes
    float local_sum = 0.0f;
    for (int i = tid; i < dim; i += stride) {
        float val = expf(row_ptr[i] - max_val);
        local_sum += val;
        out_ptr[i] = val;
    }
    
    // Warp-level reduction
    local_sum = warp_reduce_sum(local_sum);
    
    // Cross-warp reduction
    if (lane_id == 0 && warp_id < 16) {
        smem[warp_id] = local_sum;
    }
    __syncthreads();
    
    float sum_exp;
    if (tid < 16) {
        sum_exp = smem[tid];
        sum_exp = warp_reduce_sum(sum_exp);
        if (tid == 0) smem[0] = sum_exp;
    }
    __syncthreads();
    sum_exp = smem[0] + 1e-7f;
    
    // Phase 3: Normalize
    float inv_sum = 1.0f / sum_exp;
    for (int i = tid; i < dim; i += stride) {
        out_ptr[i] *= inv_sum;
    }
}

torch::Tensor softmax_hip(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 2, "Input must be 2D tensor");
    int batch_size = x.size(0);
    int dim = x.size(1);
    
    auto out = torch::empty_like(x);
    
    const int block_size = 512;  // Increased for better occupancy
    dim3 grid(batch_size);
    dim3 block(block_size);
    
    hipLaunchKernelGGL(
        softmax_kernel,
        grid, block, 0, 0,
        x.data_ptr<float>(), out.data_ptr<float>(), batch_size, dim
    );
    
    return out;
}
"""

softmax = load_inline(
    name="softmax_optimized",
    cpp_sources=softmax_hip_source,
    functions=["softmax_hip"],
    verbose=True,
    with_cuda=True,
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom HIP softmax kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax = softmax
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation using custom HIP kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return self.softmax.softmax_hip(x)