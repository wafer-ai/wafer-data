import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

cross_entropy_source = """
#include <hip/hip_runtime.h>
#include <cmath>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define NUM_WARPS 8

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down(val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Each thread block processes one batch item
__global__ void cross_entropy_forward_kernel(
    const float* __restrict__ predictions,
    const long* __restrict__ targets,
    float* __restrict__ loss_per_sample,
    int batch_size,
    int num_classes) {
    
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // Get target class for this batch item
    int target_class = static_cast<int>(targets[batch_idx]);
    float target_pred = predictions[batch_idx * num_classes + target_class];
    
    // First pass: find max over all classes
    float local_max = -INFINITY;
    for (int i = tid; i < num_classes; i += BLOCK_SIZE) {
        float pred = predictions[batch_idx * num_classes + i];
        local_max = fmaxf(local_max, pred);
    }
    
    // Warp-level reduction for max
    float warp_max = warp_reduce_max(local_max);
    
    // Store warp max in shared memory
    __shared__ float warp_max_values[NUM_WARPS];
    if (lane_id == 0) {
        warp_max_values[warp_id] = warp_max;
    }
    __syncthreads();
    
    // One thread computes global max
    float global_max = -INFINITY;
    if (tid == 0) {
        for (int i = 0; i < NUM_WARPS; i++) {
            global_max = fmaxf(global_max, warp_max_values[i]);
        }
        // Store for broadcasting
        warp_max_values[0] = global_max;
    }
    __syncthreads();
    global_max = warp_max_values[0];
    
    // Second pass: compute exp sum
    local_max = 0.0f;
    for (int i = tid; i < num_classes; i += BLOCK_SIZE) {
        float pred = predictions[batch_idx * num_classes + i];
        local_max += expf(pred - global_max);
    }
    
    // Warp-level reduction for sum
    float warp_sum = warp_reduce_sum(local_max);
    
    // One thread computes final loss
    if (tid == 0) {
        loss_per_sample[batch_idx] = logf(warp_sum) - (target_pred - global_max);
    }
}

// Sum losses across batch and divide by batch size
__global__ void sum_loss_kernel(
    const float* __restrict__ loss_per_sample,
    float* __restrict__ total_loss,
    int batch_size) {
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE + tid;
    
    // Grid-stride loop
    float sum = 0.0f;
    for (int i = idx; i < batch_size; i += gridDim.x * BLOCK_SIZE) {
        sum += loss_per_sample[i];
    }
    
    // Block-level reduction
    __shared__ float block_sum[BLOCK_SIZE];
    block_sum[tid] = sum;
    __syncthreads();
    
    for (int offset = BLOCK_SIZE/2; offset > 0; offset /= 2) {
        if (tid < offset) {
            block_sum[tid] += block_sum[tid + offset];
        }
        __syncthreads();
    }
    
    // Add to atomic counter
    if (tid == 0) {
        atomicAdd(total_loss, block_sum[0] / batch_size);
    }
}

torch::Tensor cross_entropy_hip(torch::Tensor predictions, torch::Tensor targets) {
    // Ensure on GPU and contiguous
    predictions = predictions.cuda().contiguous();
    targets = targets.cuda().contiguous().to(torch::kLong);
    
    auto batch_size = predictions.size(0);
    auto num_classes = predictions.size(1);
    
    // Allocate intermediate storage
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto loss_per_sample = torch::zeros({batch_size}, options);
    
    // Create a 0-dimensional tensor (scalar) for the final result
    auto total_loss = torch::zeros({}, options);  // Empty braces for 0-dim
    
    // One block per batch item
    dim3 grid_dim(batch_size);
    dim3 block_dim(BLOCK_SIZE);
    
    // Launch kernel
    hipLaunchKernelGGL(
        cross_entropy_forward_kernel,
        grid_dim, block_dim, 0, 0,
        predictions.data_ptr<float>(),
        targets.data_ptr<long>(),
        loss_per_sample.data_ptr<float>(),
        batch_size,
        num_classes
    );
    
    // Launch reduction kernel with more blocks for better parallelism
    const int sum_blocks = std::min(256, (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    hipLaunchKernelGGL(
        sum_loss_kernel,
        dim3(sum_blocks), block_dim, 0, 0,
        loss_per_sample.data_ptr<float>(),
        total_loss.data_ptr<float>(),
        batch_size
    );
    
    return total_loss;
}
"""

cross_entropy = load_inline(
    name="cross_entropy",
    cpp_sources=cross_entropy_source,
    functions=["cross_entropy_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cross_entropy = cross_entropy

    def forward(self, predictions, targets):
        return self.cross_entropy.cross_entropy_hip(predictions, targets)
