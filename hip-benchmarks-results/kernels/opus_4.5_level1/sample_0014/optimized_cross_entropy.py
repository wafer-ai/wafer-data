import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

cross_entropy_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

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
// Uses vectorized float4 loads for better memory bandwidth
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
    __shared__ float shared_max[8];  // Max 8 warps for 512 threads
    __shared__ float shared_sum[8];
    
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = BLOCK_SIZE / WARP_SIZE;
    
    // Phase 1: Find max value using vectorized loads
    float local_max = -INFINITY;
    
    // Handle float4 aligned portion
    int vec4_count = num_classes / 4;
    const float4* row_vec4 = reinterpret_cast<const float4*>(row);
    
    for (int i = tid; i < vec4_count; i += BLOCK_SIZE) {
        float4 val = row_vec4[i];
        local_max = fmaxf(local_max, val.x);
        local_max = fmaxf(local_max, val.y);
        local_max = fmaxf(local_max, val.z);
        local_max = fmaxf(local_max, val.w);
    }
    
    // Handle remainder
    for (int i = vec4_count * 4 + tid; i < num_classes; i += BLOCK_SIZE) {
        local_max = fmaxf(local_max, row[i]);
    }
    
    // Warp-level reduction for max
    local_max = warp_reduce_max(local_max);
    
    if (lane_id == 0) {
        shared_max[warp_id] = local_max;
    }
    __syncthreads();
    
    // First warp reduces across warps
    float global_max;
    if (tid < num_warps) {
        local_max = shared_max[tid];
    } else {
        local_max = -INFINITY;
    }
    if (tid < WARP_SIZE) {
        local_max = warp_reduce_max(local_max);
    }
    if (tid == 0) {
        shared_max[0] = local_max;
    }
    __syncthreads();
    global_max = shared_max[0];
    
    // Phase 2: Compute sum of exp(x - max) using vectorized loads
    float local_sum = 0.0f;
    
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
        shared_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    // First warp reduces across warps
    if (tid < num_warps) {
        local_sum = shared_sum[tid];
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
__global__ void reduce_mean_kernel(
    const float* __restrict__ losses,
    float* __restrict__ output,
    int n
) {
    __shared__ float shared_sum[256];
    int tid = threadIdx.x;
    
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        local_sum += losses[i];
    }
    
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    // Reduction in shared memory
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
    
    // Reduce to compute mean
    reduce_mean_kernel<<<1, 256>>>(
        losses.data_ptr<float>(),
        output.data_ptr<float>(),
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
