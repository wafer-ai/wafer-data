import os
import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fuse AvgPool + GELU + Scale + Max operations
fused_kernel_cpp_source = """
#include <hip/hip_runtime.h>

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
}

__global__ void fused_avgpool_gelu_scale_max_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int features,
    int pool_kernel_size,
    float scale_factor) {
    
    extern __shared__ float shared_data[];
    
    int batch_idx = blockIdx.x;
    int pooled_features = features / pool_kernel_size;
    
    // Each thread processes one batch and a subset of pooled features
    for (int pf = threadIdx.x; pf < pooled_features; pf += blockDim.x) {
        float pool_sum = 0.0f;
        int feat_start = pf * pool_kernel_size;
        
        // Compute average pooling
        for (int i = 0; i < pool_kernel_size; i++) {
            int feat_idx = feat_start + i;
            if (feat_idx < features) {
                pool_sum += input[batch_idx * features + feat_idx];
            }
        }
        pool_sum = pool_sum / (float)pool_kernel_size;
        
        // Apply GELU and scale
        float val = gelu(pool_sum) * scale_factor;
        
        // Store for max reduction
        shared_data[pf] = val;
    }
    
    __syncthreads();
    
    // Parallel reduction to find max
    int active_threads = min(blockDim.x, pooled_features);
    
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        __syncthreads();
        if (threadIdx.x < stride && (threadIdx.x + stride) < active_threads) {
            shared_data[threadIdx.x] = fmaxf(shared_data[threadIdx.x], shared_data[threadIdx.x + stride]);
        }
    }
    
    __syncthreads();
    
    // Thread 0 writes the max value for this batch
    if (threadIdx.x == 0) {
        output[batch_idx] = shared_data[0];
    }
}

torch::Tensor fused_avgpool_gelu_scale_max_hip(
    torch::Tensor input,
    float scale_factor,
    int pool_kernel_size) {
    
    int batch_size = input.size(0);
    int features = input.size(1);
    int pooled_features = features / pool_kernel_size;
    
    auto output = torch::empty({batch_size}, input.options());
    
    int threads = 1024;
    int smem_size = pooled_features * sizeof(float);
    
    fused_avgpool_gelu_scale_max_kernel<<<batch_size, threads, smem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        features,
        pool_kernel_size,
        scale_factor);
    
    return output;
}
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_kernel_cpp_source,
    functions=["fused_avgpool_gelu_scale_max_hip"],
    verbose=True
)


class ModelNew(nn.Module):
    """
    Optimized model with fused HIP/ROCm kernel for AvgPool + GELU + Scale + Max.
    The matmul uses PyTorch's optimized implementation.
    """
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor
        
        # Use nn.Linear for matmul (already optimized)
        self.matmul = nn.Linear(in_features, out_features)
        
        # Fused operations kernel
        self.fused_ops = fused_ops

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        # Optimized matmul using PyTorch's implementation
        x = self.matmul(x)
        
        # Apply fused AvgPool + GELU + Scale + Max
        # Note: AvgPool1d adds/removes a dimension, we handle the reshape ourselves
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = x.view(x.size(0), x.size(2))  # (batch, features)
        
        output = self.fused_ops.fused_avgpool_gelu_scale_max_hip(
            x, self.scale_factor, self.pool_kernel_size
        )
        
        return output


# Keep the same interface for initialization
batch_size = 1024
in_features = 8192
out_features = 8192
pool_kernel_size = 16
scale_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, pool_kernel_size, scale_factor]