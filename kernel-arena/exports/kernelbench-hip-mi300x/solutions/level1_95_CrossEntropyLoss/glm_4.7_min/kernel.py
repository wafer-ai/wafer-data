import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

cross_entropy_cpp_source = """
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

__global__ void fused_cross_entropy_kernel(
    const float* __restrict__ predictions,
    const int64_t* __restrict__ targets,
    float* __restrict__ output,
    int batch_size,
    int num_classes
) {
    // Each thread block processes multiple rows for better occupancy
    int block_id = blockIdx.x;
    int rows_per_block = (batch_size + gridDim.x - 1) / gridDim.x;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, batch_size);
    
    // Shared memory for reduction
    extern __shared__ float sdata[];
    
    // Process each row assigned to this block
    for (int row = row_start; row < row_end; row++) {
        const float* row_pred = predictions + row * num_classes;
        int target = targets[row];
        
        // Phase 1: Find max for numerical stability
        float max_val = -INFINITY;
        
        // Each thread processes multiple elements
        int tid = threadIdx.x;
        for (int c = tid; c < num_classes; c += blockDim.x) {
            float val = row_pred[c];
            if (val > max_val) {
                max_val = val;
            }
        }
        
        // Reduce max across threads
        sdata[tid] = max_val;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
            }
            __syncthreads();
        }
        max_val = sdata[0];
        __syncthreads();
        
        // Phase 2: Compute sum of exp(x - max) and look up target contribution
        float sum_exp = 0.0f;
        float target_exp = 0.0f;  // exp(x[target] - max)
        
        for (int c = tid; c < num_classes; c += blockDim.x) {
            float exp_val = expf(row_pred[c] - max_val);
            sum_exp += exp_val;
            if (c == target) {
                target_exp = exp_val;
            }
        }
        
        // Reduce sum_exp across threads and broadcast target_exp
        float target_exp_local = target_exp;
        sdata[tid] = sum_exp;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        sum_exp = sdata[0];
        
        // Phase 3: Combine target_exp across threads (in case target was processed by a thread)
        // Actually, we need to be careful here - only one thread found the target value
        // Let's use atomics or just do a simple reduction for target_exp
        sdata[tid] = target_exp_local;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];  // Only one will be non-zero
            }
            __syncthreads();
        }
        target_exp = sdata[0];
        
        // Phase 4: Compute loss in first thread
        // CE = -log(softmax[target]) = -log(exp(x[target] - max) / sum_exp)
        //    = -(x[target] - max) - log(sum_exp) = -target_exp/sum_exp... wait
        // Actually: -log(target_exp / sum_exp) = -target_exp + log(sum_exp)? NO
        // CE = -log(target_exp / sum_exp) = -log(target_exp) + log(sum_exp)
        //    = -(x[target] - max) + log(sum_exp)
        //    = -x[target] + max + log(sum_exp)
        if (tid == 0) {
            // Handle numerical edge case
            if (sum_exp > 0.0f && target_exp > 0.0f) {
                // Use the direct formula for stability
                output[row] = -logf(target_exp / sum_exp);
            } else {
                output[row] = 0.0f;  // Should not happen with valid inputs
            }
        }
    }
}

torch::Tensor fused_cross_entropy_hip(torch::Tensor predictions, torch::Tensor targets) {
    auto batch_size = predictions.size(0);
    auto num_classes = predictions.size(1);
    auto output = torch::empty({batch_size}, predictions.options());
    
    const int block_size = 256;
    const int num_blocks = min(8192, (batch_size + 3) / 4);  // Limit blocks for better scheduling
    
    size_t shared_mem_size = block_size * sizeof(float);
    
    fused_cross_entropy_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        batch_size,
        num_classes
    );
    
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::stringstream ss;
        ss << "HIP kernel launch failed: " << hipGetErrorString(err);
        throw std::runtime_error(ss.str());
    }
    
    return output;
}

torch::Tensor fused_cross_entropy_mean_hip(torch::Tensor predictions, torch::Tensor targets) {
    auto loss_per_sample = fused_cross_entropy_hip(predictions, targets);
    return loss_per_sample.mean();
}
"""

cross_entropy = load_inline(
    name="cross_entropy_loss",
    cpp_sources=cross_entropy_cpp_source,
    functions=["fused_cross_entropy_hip", "fused_cross_entropy_mean_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized Cross Entropy Loss model using fused HIP kernel.
    Computes softmax and negative log-likelihood in a single kernel launch.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cross_entropy = cross_entropy

    def forward(self, predictions, targets):
        # Use fused kernel for better performance
        return self.cross_entropy.fused_cross_entropy_mean_hip(predictions, targets)


def get_inputs():
    batch_size = 32768
    num_classes = 4096
    input_shape = (num_classes,)
    dim = 1
    return [torch.rand(batch_size, *input_shape).cuda(), torch.randint(0, num_classes, (batch_size,)).cuda()]


def get_init_inputs():
    return []