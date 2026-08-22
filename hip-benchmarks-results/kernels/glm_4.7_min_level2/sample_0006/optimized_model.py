import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

max_pool_sum_scale_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void max_pool_sum_scale_kernel(const float* x, float* out, int batch_size, int features, float scale) {
    // Each thread block processes one batch element
    // Each thread within the block starts with one pooled value (from max pool of 2)
    // Then we reduce within the block to sum all pooled values
    
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    int pooled_features = features / 2;  // After max pooling with kernel size 2
    
    __shared__ float sdata[512];
    
    int pooled_idx = threadIdx.x;
    
    // Each thread handles one pooled feature
    // The pooled value is max of two consecutive features
    float local_val;
    if (pooled_idx < pooled_features) {
        int idx0 = batch_idx * features + pooled_idx * 2;
        int idx1 = batch_idx * features + pooled_idx * 2 + 1;
        local_val = max(x[idx0], x[idx1]);
    } else {
        local_val = 0.0f;
    }
    
    sdata[threadIdx.x] = local_val;
    __syncthreads();
    
    // Reduction: sum all pooled values for this batch element
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Thread 0 writes the scaled sum
    if (threadIdx.x == 0) {
        out[batch_idx] = sdata[0] * scale;
    }
}

torch::Tensor max_pool_sum_scale_hip(torch::Tensor x, float scale) {
    int batch_size = x.size(0);
    int features = x.size(1);
    int pooled_features = features / 2;
    
    auto out = torch::zeros({batch_size}, x.options());
    
    const int block_size = 512;
    int num_blocks = batch_size;
    
    max_pool_sum_scale_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        batch_size, 
        features, 
        scale
    );
    
    return out;
}
"""

max_pool_sum_scale = load_inline(
    name="max_pool_sum_scale",
    cpp_sources=max_pool_sum_scale_cpp_source,
    functions=["max_pool_sum_scale_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.max_pool_sum_scale = max_pool_sum_scale
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        # Keep the optimized cuBLAS/rocBLAS matmul
        x = self.matmul(x)
        
        # Fuse MaxPool + Sum + Scale into a single kernel
        x = self.max_pool_sum_scale.max_pool_sum_scale_hip(x, self.scale_factor)
        
        return x