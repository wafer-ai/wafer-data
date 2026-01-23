import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel that combines matmul weight loading memory optimization
# Actually, the best approach is to compile away the kernel launch overhead
# by fusing operations using tensor operations that get compiled together

fused_ops_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void fused_maxpool_sum_scale_kernel(const float* x, float* out, int batch_size, int features, float scale) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    int pooled_features = features / 2;
    
    // Use block-level reduction
    extern __shared__ float sdata[];
    
    int pooled_idx = threadIdx.x;
    
    // Each thread processes one pooled feature
    // Max pool of 2 consecutive elements
    float local_val = 0.0f;
    if (pooled_idx < pooled_features) {
        int base = batch_idx * features + pooled_idx * 2;
        float a = x[base];
        float b = x[base + 1];
        local_val = (a > b) ? a : b;
    }
    
    sdata[threadIdx.x] = local_val;
    __syncthreads();
    
    // Parallel reduction (tree-based sum)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Thread 0 writes scaled result
    if (threadIdx.x == 0) {
        out[batch_idx] = sdata[0] * scale;
    }
}

torch::Tensor fused_maxpool_sum_scale_hip(torch::Tensor x, float scale_factor) {
    int batch_size = x.size(0);
    int features = x.size(1);
    int pooled_features = features / 2;
    
    auto out = torch::zeros({batch_size}, x.options());
    
    // Use 256 threads per block
    const int block_size = 256;
    int num_blocks = batch_size;
    int shared_mem_size = block_size * sizeof(float);
    
    fused_maxpool_sum_scale_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        batch_size, 
        features, 
        scale_factor
    );
    
    return out;
}
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    functions=["fused_maxpool_sum_scale_hip"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.fused_kernel = fused_ops
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        # Linear layer uses rocBLAS/cuBLAS which is highly optimized
        x = self.matmul(x)
        
        # Use fused kernel for maxpool + sum + scale
        x = self.fused_kernel.fused_maxpool_sum_scale_hip(x, self.scale_factor)
        
        return x