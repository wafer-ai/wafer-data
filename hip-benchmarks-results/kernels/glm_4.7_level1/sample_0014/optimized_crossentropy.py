import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

cross_entropy_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void cross_entropy_kernel(
    const float* __restrict__ predictions,
    const int64_t* __restrict__ targets,
    float* __restrict__ loss,
    int batch_size,
    int num_classes
) {
    extern __shared__ float sdata[];
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32;
    
    if (batch_idx >= batch_size) {
        sdata[threadIdx.x] = 0.0f;
        __syncthreads();
        return;
    }
    
    const float* preds = predictions + batch_idx * num_classes;
    int target_idx = targets[batch_idx];
    float target_pred = preds[target_idx];
    
    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int c = lane_id; c < num_classes; c += 32) {
        max_val = fmaxf(max_val, preds[c]);
    }
    
    // Warp reduce max
    max_val = max(max_val, __shfl_down(max_val, 16));
    max_val = max(max_val, __shfl_down(max_val, 8));
    max_val = max(max_val, __shfl_down(max_val, 4));
    max_val = max(max_val, __shfl_down(max_val, 2));
    max_val = max(max_val, __shfl_down(max_val, 1));
    max_val = __shfl(max_val, 0);
    
    // Compute sum of exp
    float sum_exp = 0.0f;
    for (int c = lane_id; c < num_classes; c += 32) {
        sum_exp += expf(preds[c] - max_val);
    }
    
    // Warp reduce sum
    sum_exp += __shfl_down(sum_exp, 16);
    sum_exp += __shfl_down(sum_exp, 8);
    sum_exp += __shfl_down(sum_exp, 4);
    sum_exp += __shfl_down(sum_exp, 2);
    sum_exp += __shfl_down(sum_exp, 1);
    sum_exp = __shfl(sum_exp, 0);
    
    // Cross entropy for this sample
    float ce = -target_pred + max_val + logf(sum_exp);
    
    sdata[threadIdx.x] = ce;
    __syncthreads();
    
    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Atomic add to global loss
    if (threadIdx.x == 0) {
        atomicAdd(loss, sdata[0]);
    }
}

torch::Tensor cross_entropy_hip(torch::Tensor predictions, torch::Tensor targets) {
    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    
    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;
    
    auto loss = torch::zeros({1}, predictions.options());
    loss.fill_(0.0f);
    
    cross_entropy_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        loss.data_ptr<float>(),
        batch_size,
        num_classes
    );
    
    // Divide by batch_size to get mean
    return loss.div_(batch_size);
}
"""

cross_entropy_module = load_inline(
    name="cross_entropy_loss",
    cpp_sources=cross_entropy_cpp_source,
    functions=["cross_entropy_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    A model that computes Cross Entropy Loss for multi-class classification tasks
    using an optimized HIP kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cross_entropy = cross_entropy_module

    def forward(self, predictions, targets):
        return self.cross_entropy.cross_entropy_hip(predictions, targets)