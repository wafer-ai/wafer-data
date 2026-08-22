import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

cross_entropy_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>
#include <limits>
#include <cfloat>

#define WARP_SIZE 64
#define BLOCK_SIZE 1024  // Increased threads per block

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

// Online softmax cross entropy kernel - single pass through data
// Uses online algorithm to compute max and sum simultaneously for better cache usage
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
    __shared__ float shared_max[NUM_WARPS];
    __shared__ float shared_sum[NUM_WARPS];
    
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // Online softmax: track running max and correction factor
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    
    // Vectorized float4 loads for better memory bandwidth
    int vec4_count = num_classes / 4;
    const float4* row_vec4 = reinterpret_cast<const float4*>(row);
    
    // Single pass through data using online algorithm
    #pragma unroll 4
    for (int i = tid; i < vec4_count; i += BLOCK_SIZE) {
        float4 val = row_vec4[i];
        
        // Process each element with online update
        float new_max = fmaxf(fmaxf(fmaxf(fmaxf(local_max, val.x), val.y), val.z), val.w);
        if (new_max > local_max) {
            local_sum = local_sum * expf(local_max - new_max);
            local_max = new_max;
        }
        local_sum += expf(val.x - local_max) + expf(val.y - local_max) + 
                     expf(val.z - local_max) + expf(val.w - local_max);
    }
    
    // Handle remainder
    for (int i = vec4_count * 4 + tid; i < num_classes; i += BLOCK_SIZE) {
        float val = row[i];
        float new_max = fmaxf(local_max, val);
        if (new_max > local_max) {
            local_sum = local_sum * expf(local_max - new_max);
            local_max = new_max;
        }
        local_sum += expf(val - local_max);
    }
    
    // Warp-level reduction combining max and sum
    // First reduce max within warp
    float warp_max = warp_reduce_max(local_max);
    
    // Rescale local_sum to global max within warp
    local_sum = local_sum * expf(local_max - warp_max);
    
    // Now reduce sum within warp
    float warp_sum = warp_reduce_sum(local_sum);
    
    // First lane of each warp writes to shared memory
    if (lane_id == 0) {
        shared_max[warp_id] = warp_max;
        shared_sum[warp_id] = warp_sum;
    }
    __syncthreads();
    
    // Final reduction across warps
    float global_max, global_sum;
    if (tid < NUM_WARPS) {
        local_max = shared_max[tid];
        local_sum = shared_sum[tid];
    } else {
        local_max = -FLT_MAX;
        local_sum = 0.0f;
    }
    
    if (tid < WARP_SIZE) {
        float final_max = warp_reduce_max(local_max);
        float rescaled_sum = local_sum * expf(local_max - final_max);
        float final_sum = warp_reduce_sum(rescaled_sum);
        
        if (tid == 0) {
            shared_max[0] = final_max;
            shared_sum[0] = final_sum;
        }
    }
    __syncthreads();
    
    global_max = shared_max[0];
    global_sum = shared_sum[0];
    
    // Compute final loss: -x[target] + max + log(sum)
    if (tid == 0) {
        float log_sum_exp = global_max + logf(global_sum);
        float target_val = row[target];
        losses[batch_idx] = log_sum_exp - target_val;
    }
}

// Hierarchical reduction kernel to compute mean of losses
// Optimized for large batch sizes
__global__ void reduce_mean_kernel(
    const float* __restrict__ losses,
    float* __restrict__ output,
    int n
) {
    __shared__ float shared_sum[256];
    int tid = threadIdx.x;
    
    // Vectorized loads
    float local_sum = 0.0f;
    int vec4_count = n / 4;
    const float4* losses_vec4 = reinterpret_cast<const float4*>(losses);
    
    for (int i = tid; i < vec4_count; i += blockDim.x) {
        float4 v = losses_vec4[i];
        local_sum += v.x + v.y + v.z + v.w;
    }
    
    // Handle remainder
    for (int i = vec4_count * 4 + tid; i < n; i += blockDim.x) {
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
    extra_cuda_cflags=["-O3"]
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
