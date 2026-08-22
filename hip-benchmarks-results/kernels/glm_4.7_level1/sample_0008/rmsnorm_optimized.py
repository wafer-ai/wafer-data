import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

rmsnorm_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void rmsnorm_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const int batch_size,
    const int features,
    const int dim1,
    const int dim2,
    const float eps) {
    
    // Each thread block processes one (batch, d1, d2) coordinate
    // Within the block, threads reduce over the feature dimension
    
    int64_t coord_idx = blockIdx.x;
    int64_t total_coords = (int64_t)batch_size * dim1 * dim2;
    
    if (coord_idx >= total_coords) return;
    
    // Compute (batch, d1, d2) from linear index
    int tmp = coord_idx;
    int d2 = tmp % dim2;
    tmp /= dim2;
    int d1 = tmp % dim1;
    int batch = tmp / dim1;
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Shared memory for reduction
    __shared__ float s_data[256];
    
    // Compute sum of squares for features assigned to this thread
    float thread_sum = 0.0f;
    for (int f = tid; f < features; f += block_size) {
        int64_t idx = (batch * (int64_t)dim1 * dim2 * features) + 
                       (f * dim1 * dim2) + 
                       (d1 * dim2) + d2;
        float val = x[idx];
        thread_sum += val * val;
    }
    
    s_data[tid] = thread_sum;
    __syncthreads();
    
    // Parallel reduction following CUDA sample pattern
    for (int stride = block_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 computes RMS and stores to shared[0]
    if (tid == 0) {
        s_data[0] = sqrtf(s_data[0] / features + eps);
    }
    __syncthreads();
    
    float rms = s_data[0];
    
    // Write outputs using same pattern as reduction
    for (int f = tid; f < features; f += block_size) {
        int64_t idx = (batch * (int64_t)dim1 * dim2 * features) + 
                       (f * dim1 * dim2) + 
                       (d1 * dim2) + d2;
        out[idx] = x[idx] / rms;
    }
}

torch::Tensor rmsnorm_hip(torch::Tensor x, int num_features, float eps) {
    auto batch_size = x.size(0);
    auto features = x.size(1);
    auto dim1 = x.size(2);
    auto dim2 = x.size(3);
    
    auto out = torch::zeros_like(x);
    
    int64_t total_coords = (int64_t)batch_size * dim1 * dim2;
    
    // Use 256 threads per block (each block processes one coordinate)
    const int block_size = 256;
    int num_blocks = total_coords;
    
    rmsnorm_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        features,
        dim1,
        dim2,
        eps
    );
    
    return out;
}
"""

rmsnorm_module = load_inline(
    name="rmsnorm_module",
    cpp_sources=rmsnorm_cpp_source,
    functions=["rmsnorm_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Simple model that performs RMS Normalization with optimized HIP kernel.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.rmsnorm = rmsnorm_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor using optimized HIP kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, dim1, dim2).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        return self.rmsnorm.rmsnorm_hip(x, self.num_features, self.eps)