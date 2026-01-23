import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

softmax_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <float.h>

#define WARP_SIZE 64
#define NUM_WARPS 16
#define BLOCK_SIZE (WARP_SIZE * NUM_WARPS)  // 1024 threads

// Online softmax warp reduction
__device__ __forceinline__ void warp_reduce_online(float& max_val, float& sum_val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_max = __shfl_xor(max_val, offset);
        float other_sum = __shfl_xor(sum_val, offset);
        
        float new_max = fmaxf(max_val, other_max);
        sum_val = sum_val * expf(max_val - new_max) + other_sum * expf(other_max - new_max);
        max_val = new_max;
    }
}

// Kernel 1: Compute partial max and sum for each block
__global__ void softmax_partial_reduce(const float* __restrict__ input,
                                        float* __restrict__ partial_max,
                                        float* __restrict__ partial_sum,
                                        int num_rows, int num_cols, int blocks_per_row) {
    extern __shared__ char smem[];
    float* s_max = reinterpret_cast<float*>(smem);
    float* s_sum = s_max + NUM_WARPS;
    
    int row = blockIdx.x / blocks_per_row;
    int block_in_row = blockIdx.x % blocks_per_row;
    if (row >= num_rows) return;
    
    const float* row_in = input + row * num_cols;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // Each block handles a portion of the row
    int cols_per_block = (num_cols + blocks_per_row - 1) / blocks_per_row;
    int start_col = block_in_row * cols_per_block;
    int end_col = min(start_col + cols_per_block, num_cols);
    
    // Online reduction within this block's portion
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    
    for (int i = start_col + tid; i < end_col; i += BLOCK_SIZE) {
        float val = row_in[i];
        float new_max = fmaxf(local_max, val);
        local_sum = local_sum * expf(local_max - new_max) + expf(val - new_max);
        local_max = new_max;
    }
    
    // Warp reduction
    warp_reduce_online(local_max, local_sum);
    
    if (lane_id == 0) {
        s_max[warp_id] = local_max;
        s_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    // Final reduction within block
    if (tid < NUM_WARPS) {
        local_max = s_max[tid];
        local_sum = s_sum[tid];
    } else {
        local_max = -FLT_MAX;
        local_sum = 0.0f;
    }
    
    if (tid < WARP_SIZE) {
        warp_reduce_online(local_max, local_sum);
    }
    
    // Write partial results
    if (tid == 0) {
        int idx = row * blocks_per_row + block_in_row;
        partial_max[idx] = local_max;
        partial_sum[idx] = local_sum;
    }
}

// Kernel 2: Reduce partial results and compute final softmax
__global__ void softmax_final(const float* __restrict__ input,
                               float* __restrict__ output,
                               const float* __restrict__ partial_max,
                               const float* __restrict__ partial_sum,
                               int num_rows, int num_cols, int blocks_per_row) {
    extern __shared__ char smem[];
    float* s_max = reinterpret_cast<float*>(smem);
    float* s_sum = s_max + NUM_WARPS;
    
    int row = blockIdx.x / blocks_per_row;
    int block_in_row = blockIdx.x % blocks_per_row;
    if (row >= num_rows) return;
    
    const float* row_in = input + row * num_cols;
    float* row_out = output + row * num_cols;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // Reduce partial results to get global max and sum
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    
    for (int i = tid; i < blocks_per_row; i += BLOCK_SIZE) {
        int idx = row * blocks_per_row + i;
        float pmax = partial_max[idx];
        float psum = partial_sum[idx];
        
        float new_max = fmaxf(local_max, pmax);
        local_sum = local_sum * expf(local_max - new_max) + psum * expf(pmax - new_max);
        local_max = new_max;
    }
    
    warp_reduce_online(local_max, local_sum);
    
    if (lane_id == 0) {
        s_max[warp_id] = local_max;
        s_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    if (tid < NUM_WARPS) {
        local_max = s_max[tid];
        local_sum = s_sum[tid];
    } else {
        local_max = -FLT_MAX;
        local_sum = 0.0f;
    }
    
    if (tid < WARP_SIZE) {
        warp_reduce_online(local_max, local_sum);
    }
    
    if (tid == 0) {
        s_max[0] = local_max;
        s_sum[0] = local_sum;
    }
    __syncthreads();
    
    float row_max = s_max[0];
    float inv_sum = 1.0f / s_sum[0];
    
    // Compute and write this block's portion of the output
    int cols_per_block = (num_cols + blocks_per_row - 1) / blocks_per_row;
    int start_col = block_in_row * cols_per_block;
    int end_col = min(start_col + cols_per_block, num_cols);
    
    for (int i = start_col + tid; i < end_col; i += BLOCK_SIZE) {
        row_out[i] = expf(row_in[i] - row_max) * inv_sum;
    }
}

// Single-kernel online softmax for comparison
__global__ void softmax_kernel_online_single(const float* __restrict__ input, 
                                              float* __restrict__ output,
                                              int num_rows, int num_cols) {
    extern __shared__ char shared_mem[];
    float* s_max = reinterpret_cast<float*>(shared_mem);
    float* s_sum = s_max + NUM_WARPS;
    
    int row = blockIdx.x;
    if (row >= num_rows) return;
    
    const float* row_in = input + row * num_cols;
    float* row_out = output + row * num_cols;
    
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // Online softmax reduction
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    
    for (int i = tid; i < num_cols; i += BLOCK_SIZE) {
        float val = row_in[i];
        float new_max = fmaxf(local_max, val);
        local_sum = local_sum * expf(local_max - new_max) + expf(val - new_max);
        local_max = new_max;
    }
    
    warp_reduce_online(local_max, local_sum);
    
    if (lane_id == 0) {
        s_max[warp_id] = local_max;
        s_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    if (tid < NUM_WARPS) {
        local_max = s_max[tid];
        local_sum = s_sum[tid];
    } else {
        local_max = -FLT_MAX;
        local_sum = 0.0f;
    }
    
    if (tid < WARP_SIZE) {
        warp_reduce_online(local_max, local_sum);
    }
    
    if (tid == 0) {
        s_max[0] = local_max;
        s_sum[0] = local_sum;
    }
    __syncthreads();
    
    float row_max = s_max[0];
    float inv_sum = 1.0f / s_sum[0];
    
    for (int i = tid; i < num_cols; i += BLOCK_SIZE) {
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
    
    int shared_mem_size = 2 * NUM_WARPS * sizeof(float);
    
    // For very large rows, use multi-block approach
    if (num_cols > 100000) {
        // Use multiple blocks per row
        int blocks_per_row = 8;  // Tune this
        int total_blocks = num_rows * blocks_per_row;
        
        auto partial_max = torch::empty({num_rows, blocks_per_row}, input.options());
        auto partial_sum = torch::empty({num_rows, blocks_per_row}, input.options());
        
        // Kernel 1: Compute partial reductions
        softmax_partial_reduce<<<total_blocks, BLOCK_SIZE, shared_mem_size>>>(
            input.data_ptr<float>(),
            partial_max.data_ptr<float>(),
            partial_sum.data_ptr<float>(),
            num_rows, num_cols, blocks_per_row
        );
        
        // Kernel 2: Final reduction and output
        softmax_final<<<total_blocks, BLOCK_SIZE, shared_mem_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            partial_max.data_ptr<float>(),
            partial_sum.data_ptr<float>(),
            num_rows, num_cols, blocks_per_row
        );
    } else {
        // Single block per row for smaller rows
        softmax_kernel_online_single<<<num_rows, BLOCK_SIZE, shared_mem_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            num_rows, num_cols
        );
    }
    
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
