import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

cross_entropy_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>
#include <cfloat>

#define WARP_SIZE 64
#define BLOCK_SIZE 512

// Warp reduction for max using AMD wavefront size of 64
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

// Warp reduction for sum using AMD wavefront size of 64
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Cross entropy kernel: one block per batch element
// Two-pass algorithm with aggressive vectorization
__global__ void cross_entropy_kernel(
    const float* __restrict__ predictions,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int num_classes
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    const float* row = predictions + batch_idx * num_classes;
    int target = targets[batch_idx];
    
    // Shared memory for block-level reductions
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float shared_data[NUM_WARPS];
    
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // Phase 1: Find max value using vectorized loads
    float local_max = -FLT_MAX;
    
    // Handle float4 aligned portion
    int vec4_count = num_classes / 4;
    const float4* row_vec4 = reinterpret_cast<const float4*>(row);
    
    #pragma unroll 8
    for (int i = tid; i < vec4_count; i += BLOCK_SIZE) {
        float4 val = row_vec4[i];
        local_max = fmaxf(local_max, fmaxf(fmaxf(fmaxf(val.x, val.y), val.z), val.w));
    }
    
    // Handle remainder
    for (int i = vec4_count * 4 + tid; i < num_classes; i += BLOCK_SIZE) {
        local_max = fmaxf(local_max, row[i]);
    }
    
    // Warp-level reduction for max
    local_max = warp_reduce_max(local_max);
    
    if (lane_id == 0) {
        shared_data[warp_id] = local_max;
    }
    __syncthreads();
    
    // First warp reduces across warps
    float global_max;
    if (tid < NUM_WARPS) {
        local_max = shared_data[tid];
    } else {
        local_max = -FLT_MAX;
    }
    if (tid < WARP_SIZE) {
        local_max = warp_reduce_max(local_max);
    }
    if (tid == 0) {
        shared_data[0] = local_max;
    }
    __syncthreads();
    global_max = shared_data[0];
    
    // Phase 2: Compute sum of exp(x - max) using vectorized loads
    float local_sum = 0.0f;
    
    #pragma unroll 8
    for (int i = tid; i < vec4_count; i += BLOCK_SIZE) {
        float4 val = row_vec4[i];
        local_sum += expf(val.x - global_max);
        local_sum += expf(val.y - global_max);
        local_sum += expf(val.z - global_max);
        local_sum += expf(val.w - global_max);
    }
    
    // Handle remainder
    for (int i = vec4_count * 4 + tid; i < num_classes; i += BLOCK_SIZE) {
        local_sum += expf(row[i] - global_max);
    }
    
    // Warp-level reduction for sum
    local_sum = warp_reduce_sum(local_sum);
    
    if (lane_id == 0) {
        shared_data[warp_id] = local_sum;
    }
    __syncthreads();
    
    // First warp reduces across warps
    if (tid < NUM_WARPS) {
        local_sum = shared_data[tid];
    } else {
        local_sum = 0.0f;
    }
    if (tid < WARP_SIZE) {
        local_sum = warp_reduce_sum(local_sum);
    }
    
    // Compute final loss: -x[target] + max + log(sum)
    if (tid == 0) {
        float log_sum_exp = global_max + logf(local_sum);
        float target_val = row[target];
        losses[batch_idx] = log_sum_exp - target_val;
    }
}

// Hierarchical reduction kernel to compute mean of losses
// Uses multiple blocks for better performance on large batch
__global__ void reduce_mean_partial(
    const float* __restrict__ losses,
    float* __restrict__ partial_sums,
    int n
) {
    __shared__ float shared_sum[256];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_size = blockDim.x;
    int grid_size = gridDim.x;
    
    int elements_per_block = (n + grid_size - 1) / grid_size;
    int start = bid * elements_per_block;
    int end = min(start + elements_per_block, n);
    
    float local_sum = 0.0f;
    for (int i = start + tid; i < end; i += block_size) {
        local_sum += losses[i];
    }
    
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[bid] = shared_sum[0];
    }
}

__global__ void reduce_mean_final(
    const float* __restrict__ partial_sums,
    float* __restrict__ output,
    int num_partials,
    int n
) {
    __shared__ float shared_sum[256];
    int tid = threadIdx.x;
    
    float local_sum = 0.0f;
    for (int i = tid; i < num_partials; i += blockDim.x) {
        local_sum += partial_sums[i];
    }
    
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[0] = shared_sum[0] / (float)n;
    }
}

torch::Tensor cross_entropy_hip(torch::Tensor predictions, torch::Tensor targets) {
    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    
    auto losses = torch::empty({batch_size}, predictions.options());
    auto output = torch::empty({1}, predictions.options());
    
    // Launch cross entropy kernel - one block per batch element
    cross_entropy_kernel<<<batch_size, BLOCK_SIZE>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        num_classes
    );
    
    // Use hierarchical reduction for large batches
    int num_blocks = 128;
    auto partial_sums = torch::empty({num_blocks}, predictions.options());
    
    reduce_mean_partial<<<num_blocks, 256>>>(
        losses.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        batch_size
    );
    
    reduce_mean_final<<<1, 256>>>(
        partial_sums.data_ptr<float>(),
        output.data_ptr<float>(),
        num_blocks,
        batch_size
    );
    
    return output.squeeze();
}
"""

cross_entropy_cpp = """
torch::Tensor cross_entropy_hip(torch::Tensor predictions, torch::Tensor targets);
"""

cross_entropy_module = load_inline(
    name="cross_entropy_hip",
    cpp_sources=cross_entropy_cpp,
    cuda_sources=cross_entropy_source,
    functions=["cross_entropy_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cross_entropy = cross_entropy_module

    def forward(self, predictions, targets):
        return self.cross_entropy.cross_entropy_hip(predictions, targets)


def get_inputs():
    batch_size = 32768
    num_classes = 4096
    return [torch.rand(batch_size, num_classes).cuda(), torch.randint(0, num_classes, (batch_size,)).cuda()]


def get_init_inputs():
    return []
