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
#define THREADS_PER_BLOCK 256

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

// Fused max and sum kernel using online softmax algorithm
// Each block handles one sample
__global__ void cross_entropy_fused_kernel(
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
    __shared__ float shared_max[8];
    __shared__ float shared_d[8];  // For online softmax: d = sum(exp(x - max))
    
    // Online softmax - compute max and sum in a single pass
    // m = running max, d = sum of exp(x_i - m)
    float m = -FLT_MAX;
    float d = 0.0f;
    
    // Vectorized loading with float4
    int vec_num_classes = num_classes / 4;
    const float4* pred_row_vec = reinterpret_cast<const float4*>(pred_row);
    
    #pragma unroll 2
    for (int i = tid; i < vec_num_classes; i += block_size) {
        float4 vals = pred_row_vec[i];
        
        // Process each element with online softmax update
        float old_m = m;
        m = fmaxf(m, vals.x);
        d = d * expf(old_m - m) + expf(vals.x - m);
        
        old_m = m;
        m = fmaxf(m, vals.y);
        d = d * expf(old_m - m) + expf(vals.y - m);
        
        old_m = m;
        m = fmaxf(m, vals.z);
        d = d * expf(old_m - m) + expf(vals.z - m);
        
        old_m = m;
        m = fmaxf(m, vals.w);
        d = d * expf(old_m - m) + expf(vals.w - m);
    }
    
    // Handle remainder
    for (int i = vec_num_classes * 4 + tid; i < num_classes; i += block_size) {
        float val = pred_row[i];
        float old_m = m;
        m = fmaxf(m, val);
        d = d * expf(old_m - m) + expf(val - m);
    }
    
    // Warp-level reduction for online softmax
    // Need to combine (m, d) pairs: m_new = max(m1, m2), d_new = d1*exp(m1-m_new) + d2*exp(m2-m_new)
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_m = __shfl_xor(m, offset);
        float other_d = __shfl_xor(d, offset);
        float new_m = fmaxf(m, other_m);
        d = d * expf(m - new_m) + other_d * expf(other_m - new_m);
        m = new_m;
    }
    
    if (lane_id == 0) {
        shared_max[warp_id] = m;
        shared_d[warp_id] = d;
    }
    __syncthreads();
    
    // Final reduction across warps (only first warp)
    if (warp_id == 0) {
        m = (lane_id < num_warps) ? shared_max[lane_id] : -FLT_MAX;
        d = (lane_id < num_warps) ? shared_d[lane_id] : 0.0f;
        
        #pragma unroll
        for (int offset = num_warps / 2; offset > 0; offset /= 2) {
            float other_m = __shfl_xor(m, offset);
            float other_d = __shfl_xor(d, offset);
            float new_m = fmaxf(m, other_m);
            d = d * expf(m - new_m) + other_d * expf(other_m - new_m);
            m = new_m;
        }
        
        if (lane_id == 0) {
            shared_max[0] = m;
            shared_d[0] = d;
        }
    }
    __syncthreads();
    
    // Compute cross entropy loss for this sample
    // loss = -log(softmax(pred[target])) = -(pred[target] - max - log(sum_exp))
    if (tid == 0) {
        float final_max = shared_max[0];
        float final_sum = shared_d[0];
        float target_pred = pred_row[target];
        float log_softmax = target_pred - final_max - logf(final_sum);
        losses[sample_idx] = -log_softmax;
    }
}

// Efficient reduction for mean using parallel reduction
__global__ void reduce_mean_kernel(
    const float* __restrict__ losses,
    float* __restrict__ output,
    int batch_size
) {
    __shared__ float shared_data[256];
    
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
    
    // Launch one block per sample with 256 threads
    cross_entropy_fused_kernel<<<batch_size, THREADS_PER_BLOCK>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        num_classes,
        batch_size
    );
    
    // Reduce to get mean loss
    reduce_mean_kernel<<<1, 256>>>(
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
