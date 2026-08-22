import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

rmsnorm_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Fused RMSNorm kernel - each thread handles one spatial position
// Input shape: (batch, features, dim1, dim2)
// Normalization along features dimension (dim=1)
__global__ void rmsnorm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int dim1,
    const int dim2,
    const float eps
) {
    // Each thread processes one (batch, spatial) position
    const int spatial_size = dim1 * dim2;
    const int total_positions = batch_size * spatial_size;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_positions) return;
    
    // Convert linear index to batch and spatial indices
    int batch_idx = idx / spatial_size;
    int spatial_idx = idx % spatial_size;
    int d1 = spatial_idx / dim2;
    int d2 = spatial_idx % dim2;
    
    // Compute sum of squares across features
    float sum_sq = 0.0f;
    
    // Input is contiguous: [batch, features, dim1, dim2]
    // Stride for features dimension
    const int feature_stride = dim1 * dim2;
    const int batch_offset = batch_idx * num_features * feature_stride;
    const int spatial_offset = d1 * dim2 + d2;
    
    #pragma unroll 8
    for (int f = 0; f < num_features; f++) {
        int in_idx = batch_offset + f * feature_stride + spatial_offset;
        float val = input[in_idx];
        sum_sq += val * val;
    }
    
    // Compute RMS
    float rms = sqrtf(sum_sq / num_features + eps);
    float inv_rms = 1.0f / rms;
    
    // Normalize and write output
    #pragma unroll 8
    for (int f = 0; f < num_features; f++) {
        int in_idx = batch_offset + f * feature_stride + spatial_offset;
        output[in_idx] = input[in_idx] * inv_rms;
    }
}

torch::Tensor rmsnorm_hip(torch::Tensor input, float eps) {
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    const int dim1 = input.size(2);
    const int dim2 = input.size(3);
    
    auto output = torch::empty_like(input);
    
    const int total_positions = batch_size * dim1 * dim2;
    const int block_size = 256;
    const int num_blocks = (total_positions + block_size - 1) / block_size;
    
    rmsnorm_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_features,
        dim1,
        dim2,
        eps
    );
    
    return output;
}
"""

rmsnorm_cpp_source = """
torch::Tensor rmsnorm_hip(torch::Tensor input, float eps);
"""

rmsnorm_module = load_inline(
    name="rmsnorm_hip",
    cpp_sources=rmsnorm_cpp_source,
    cuda_sources=rmsnorm_hip_source,
    functions=["rmsnorm_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized RMS Normalization using HIP kernel.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm_module.rmsnorm_hip(x, self.eps)


def get_inputs():
    x = torch.rand(112, 64, 512, 512).cuda()
    return [x]


def get_init_inputs():
    return [64]
