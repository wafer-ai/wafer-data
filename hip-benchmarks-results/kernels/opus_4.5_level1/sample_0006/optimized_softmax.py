import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

softmax_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <float.h>

// Warp reduction for max
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 32; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

// Warp reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 32; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Online softmax kernel - each block handles one row
// Uses online algorithm to compute max and sum in single pass
__global__ void softmax_kernel(const float* __restrict__ input, 
                               float* __restrict__ output,
                               int num_rows, int num_cols) {
    extern __shared__ float smem[];
    float* s_max = smem;
    float* s_sum = smem + (blockDim.x / 64);
    
    int row = blockIdx.x;
    if (row >= num_rows) return;
    
    const float* row_in = input + row * num_cols;
    float* row_out = output + row * num_cols;
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Phase 1: Find maximum using parallel reduction
    float local_max = -FLT_MAX;
    for (int i = tid; i < num_cols; i += num_threads) {
        local_max = fmaxf(local_max, row_in[i]);
    }
    
    // Warp-level reduction for max
    local_max = warp_reduce_max(local_max);
    
    // Store warp results to shared memory
    int warp_id = tid / 64;
    int lane_id = tid % 64;
    int num_warps = num_threads / 64;
    
    if (lane_id == 0) {
        s_max[warp_id] = local_max;
    }
    __syncthreads();
    
    // Final reduction for max
    if (tid < num_warps) {
        local_max = s_max[tid];
    } else {
        local_max = -FLT_MAX;
    }
    local_max = warp_reduce_max(local_max);
    
    // Broadcast max to all threads
    if (tid == 0) {
        s_max[0] = local_max;
    }
    __syncthreads();
    float row_max = s_max[0];
    
    // Phase 2: Compute sum of exp(x - max)
    float local_sum = 0.0f;
    for (int i = tid; i < num_cols; i += num_threads) {
        local_sum += expf(row_in[i] - row_max);
    }
    
    // Warp-level reduction for sum
    local_sum = warp_reduce_sum(local_sum);
    
    if (lane_id == 0) {
        s_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    // Final reduction for sum
    if (tid < num_warps) {
        local_sum = s_sum[tid];
    } else {
        local_sum = 0.0f;
    }
    local_sum = warp_reduce_sum(local_sum);
    
    // Broadcast sum to all threads
    if (tid == 0) {
        s_sum[0] = local_sum;
    }
    __syncthreads();
    float row_sum = s_sum[0];
    
    // Phase 3: Write normalized output
    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < num_cols; i += num_threads) {
        row_out[i] = expf(row_in[i] - row_max) * inv_sum;
    }
}

torch::Tensor softmax_hip(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(input.is_cuda(), "Input must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    
    int num_rows = input.size(0);
    int num_cols = input.size(1);
    
    auto output = torch::empty_like(input);
    
    // Use 1024 threads per block for large dimensions
    int block_size = 1024;
    int num_warps = block_size / 64;
    int shared_mem_size = 2 * num_warps * sizeof(float);
    
    dim3 grid(num_rows);
    dim3 block(block_size);
    
    softmax_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_rows, num_cols
    );
    
    return output;
}
"""

softmax_cpp_source = """
torch::Tensor softmax_hip(torch::Tensor input);
"""

softmax_module = load_inline(
    name="softmax_hip",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_hip_source,
    functions=["softmax_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax_op = softmax_module
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax_op.softmax_hip(x)


def get_inputs():
    x = torch.rand(4096, 393216).cuda()
    return [x]


def get_init_inputs():
    return []
