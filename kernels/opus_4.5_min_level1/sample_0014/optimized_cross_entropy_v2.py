import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

cross_entropy_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <float.h>
#include <cmath>

#define WARP_SIZE 64
#define THREADS_PER_BLOCK 512

// Warp-level reduction for max
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

// Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Each block handles one sample (one row of predictions)
// Using online softmax algorithm for numerical stability
__global__ void cross_entropy_kernel(
    const float* __restrict__ predictions,  // [batch_size, num_classes]
    const int64_t* __restrict__ targets,    // [batch_size]
    float* __restrict__ losses,             // [batch_size] intermediate losses
    int num_classes,
    int batch_size
) {
    int sample_idx = blockIdx.x;
    if (sample_idx >= batch_size) return;
    
    const float* pred_row = predictions + sample_idx * num_classes;
    int target = targets[sample_idx];
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = block_size / WARP_SIZE;
    
    // Shared memory for inter-warp reduction
    __shared__ float shared_max[16];
    __shared__ float shared_sum[16];
    
    // Step 1: Find max value in the row using vectorized loads
    float local_max = -FLT_MAX;
    
    // Vectorized loading with float4
    int vec_num_classes = num_classes / 4;
    const float4* pred_row_vec = reinterpret_cast<const float4*>(pred_row);
    
    for (int i = tid; i < vec_num_classes; i += block_size) {
        float4 vals = pred_row_vec[i];
        local_max = fmaxf(local_max, vals.x);
        local_max = fmaxf(local_max, vals.y);
        local_max = fmaxf(local_max, vals.z);
        local_max = fmaxf(local_max, vals.w);
    }
    
    // Handle remainder
    for (int i = vec_num_classes * 4 + tid; i < num_classes; i += block_size) {
        local_max = fmaxf(local_max, pred_row[i]);
    }
    
    // Warp-level max reduction
    local_max = warp_reduce_max(local_max);
    
    if (lane_id == 0) {
        shared_max[warp_id] = local_max;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared_max[lane_id] : -FLT_MAX;
        val = warp_reduce_max(val);
        if (lane_id == 0) {
            shared_max[0] = val;
        }
    }
    __syncthreads();
    
    float max_val = shared_max[0];
    
    // Step 2: Compute sum of exp(x - max) using vectorized loads
    float local_sum = 0.0f;
    
    for (int i = tid; i < vec_num_classes; i += block_size) {
        float4 vals = pred_row_vec[i];
        local_sum += expf(vals.x - max_val);
        local_sum += expf(vals.y - max_val);
        local_sum += expf(vals.z - max_val);
        local_sum += expf(vals.w - max_val);
    }
    
    // Handle remainder
    for (int i = vec_num_classes * 4 + tid; i < num_classes; i += block_size) {
        local_sum += expf(pred_row[i] - max_val);
    }
    
    // Warp-level sum reduction
    local_sum = warp_reduce_sum(local_sum);
    
    if (lane_id == 0) {
        shared_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            shared_sum[0] = val;
        }
    }
    __syncthreads();
    
    float sum_exp = shared_sum[0];
    
    // Step 3: Compute cross entropy loss for this sample
    // loss = -log(softmax(pred[target])) = -(pred[target] - max - log(sum_exp))
    if (tid == 0) {
        float target_pred = pred_row[target];
        float log_softmax = target_pred - max_val - logf(sum_exp);
        losses[sample_idx] = -log_softmax;
    }
}

// Reduction kernel to compute mean of losses using multiple blocks
__global__ void reduce_mean_kernel(
    const float* __restrict__ losses,
    float* __restrict__ output,
    int batch_size
) {
    __shared__ float shared_data[512];
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    float local_sum = 0.0f;
    for (int i = tid; i < batch_size; i += block_size) {
        local_sum += losses[i];
    }
    
    shared_data[tid] = local_sum;
    __syncthreads();
    
    // Block reduction
    for (int stride = block_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[0] = shared_data[0] / batch_size;
    }
}

torch::Tensor cross_entropy_hip(torch::Tensor predictions, torch::Tensor targets) {
    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    
    auto losses = torch::empty({batch_size}, predictions.options());
    auto output = torch::empty({1}, predictions.options());
    
    // Launch one block per sample with 512 threads
    cross_entropy_kernel<<<batch_size, THREADS_PER_BLOCK>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        num_classes,
        batch_size
    );
    
    // Reduce to get mean loss
    reduce_mean_kernel<<<1, 512>>>(
        losses.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size
    );
    
    return output.squeeze();
}
"""

cross_entropy_cpp_source = """
torch::Tensor cross_entropy_hip(torch::Tensor predictions, torch::Tensor targets);
"""

cross_entropy_module = load_inline(
    name="cross_entropy_hip",
    cpp_sources=cross_entropy_cpp_source,
    cuda_sources=cross_entropy_hip_source,
    functions=["cross_entropy_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"],
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
