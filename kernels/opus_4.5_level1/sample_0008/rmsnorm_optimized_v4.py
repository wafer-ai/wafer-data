import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

rmsnorm_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Optimized RMSNorm kernel
// Each thread handles one spatial position, iterating over features
// Use two passes: first compute RMS, then normalize
// This allows better memory access patterns
__global__ void rmsnorm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int spatial_size,
    const float eps,
    const float inv_features
) {
    const int total_positions = batch_size * spatial_size;
    
    // Grid-stride loop for better occupancy
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total_positions; 
         idx += blockDim.x * gridDim.x) {
        
        const int batch_idx = idx / spatial_size;
        const int spatial_idx = idx % spatial_size;
        
        const int batch_offset = batch_idx * num_features * spatial_size;
        
        // Compute sum of squares - unrolled for num_features = 64
        float sum_sq = 0.0f;
        
        // Unroll by 8 for 64 features
        #pragma unroll 8
        for (int f = 0; f < 64; f++) {
            const int in_idx = batch_offset + f * spatial_size + spatial_idx;
            const float val = input[in_idx];
            sum_sq += val * val;
        }
        
        // Compute inverse RMS
        const float inv_rms = rsqrtf(sum_sq * inv_features + eps);
        
        // Normalize and write - unrolled
        #pragma unroll 8
        for (int f = 0; f < 64; f++) {
            const int in_idx = batch_offset + f * spatial_size + spatial_idx;
            output[in_idx] = input[in_idx] * inv_rms;
        }
    }
}

// General version for any num_features
__global__ void rmsnorm_kernel_general(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int spatial_size,
    const float eps,
    const float inv_features
) {
    const int total_positions = batch_size * spatial_size;
    
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total_positions; 
         idx += blockDim.x * gridDim.x) {
        
        const int batch_idx = idx / spatial_size;
        const int spatial_idx = idx % spatial_size;
        
        const int batch_offset = batch_idx * num_features * spatial_size;
        
        float sum_sq = 0.0f;
        
        for (int f = 0; f < num_features; f++) {
            const int in_idx = batch_offset + f * spatial_size + spatial_idx;
            const float val = input[in_idx];
            sum_sq += val * val;
        }
        
        const float inv_rms = rsqrtf(sum_sq * inv_features + eps);
        
        for (int f = 0; f < num_features; f++) {
            const int in_idx = batch_offset + f * spatial_size + spatial_idx;
            output[in_idx] = input[in_idx] * inv_rms;
        }
    }
}

torch::Tensor rmsnorm_hip(torch::Tensor input, float eps) {
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    const int dim1 = input.size(2);
    const int dim2 = input.size(3);
    const int spatial_size = dim1 * dim2;
    
    auto output = torch::empty_like(input);
    
    const int total_positions = batch_size * spatial_size;
    const float inv_features = 1.0f / num_features;
    
    // Use 512 threads per block for better occupancy
    const int block_size = 512;
    // Use enough blocks to saturate the GPU
    const int num_blocks = min((total_positions + block_size - 1) / block_size, 2048);
    
    if (num_features == 64) {
        rmsnorm_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            num_features,
            spatial_size,
            eps,
            inv_features
        );
    } else {
        rmsnorm_kernel_general<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            num_features,
            spatial_size,
            eps,
            inv_features
        );
    }
    
    return output;
}
"""

rmsnorm_cpp_source = """
torch::Tensor rmsnorm_hip(torch::Tensor input, float eps);
"""

rmsnorm_module = load_inline(
    name="rmsnorm_hip_v4",
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
