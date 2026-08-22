import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

rmsnorm_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Optimized RMSNorm kernel using float4 vectorization
// Each thread processes 4 spatial positions at once
__global__ void rmsnorm_vectorized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int dim1,
    const int dim2,
    const float eps
) {
    const int spatial_size = dim1 * dim2;
    const int total_positions = batch_size * spatial_size;
    const int total_vec4 = total_positions / 4;
    
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (vec_idx >= total_vec4) return;
    
    // Process 4 consecutive d2 positions
    int pos_base = vec_idx * 4;
    
    // Convert to indices
    int batch_idx = pos_base / spatial_size;
    int spatial_idx = pos_base % spatial_size;
    int d1 = spatial_idx / dim2;
    int d2_base = spatial_idx % dim2;
    
    const int feature_stride = spatial_size;
    const int batch_offset = batch_idx * num_features * feature_stride;
    
    // Initialize sums for 4 positions
    float sum_sq0 = 0.0f, sum_sq1 = 0.0f, sum_sq2 = 0.0f, sum_sq3 = 0.0f;
    
    int spatial_offset = d1 * dim2 + d2_base;
    
    // Compute sum of squares for all 4 positions
    for (int f = 0; f < num_features; f++) {
        int base_idx = batch_offset + f * feature_stride + spatial_offset;
        
        float4 vals = *reinterpret_cast<const float4*>(&input[base_idx]);
        
        sum_sq0 += vals.x * vals.x;
        sum_sq1 += vals.y * vals.y;
        sum_sq2 += vals.z * vals.z;
        sum_sq3 += vals.w * vals.w;
    }
    
    // Compute RMS inverses
    float inv_n = 1.0f / num_features;
    float inv_rms0 = rsqrtf(sum_sq0 * inv_n + eps);
    float inv_rms1 = rsqrtf(sum_sq1 * inv_n + eps);
    float inv_rms2 = rsqrtf(sum_sq2 * inv_n + eps);
    float inv_rms3 = rsqrtf(sum_sq3 * inv_n + eps);
    
    // Normalize and write output
    for (int f = 0; f < num_features; f++) {
        int base_idx = batch_offset + f * feature_stride + spatial_offset;
        
        float4 vals = *reinterpret_cast<const float4*>(&input[base_idx]);
        
        float4 out_vals;
        out_vals.x = vals.x * inv_rms0;
        out_vals.y = vals.y * inv_rms1;
        out_vals.z = vals.z * inv_rms2;
        out_vals.w = vals.w * inv_rms3;
        
        *reinterpret_cast<float4*>(&output[base_idx]) = out_vals;
    }
}

// Fallback kernel for remaining elements
__global__ void rmsnorm_scalar_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int num_features,
    const int dim1,
    const int dim2,
    const float eps,
    const int start_idx
) {
    const int spatial_size = dim1 * dim2;
    const int total_positions = batch_size * spatial_size;
    
    int idx = start_idx + blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_positions) return;
    
    int batch_idx = idx / spatial_size;
    int spatial_idx = idx % spatial_size;
    int d1 = spatial_idx / dim2;
    int d2 = spatial_idx % dim2;
    
    float sum_sq = 0.0f;
    
    const int feature_stride = spatial_size;
    const int batch_offset = batch_idx * num_features * feature_stride;
    const int spatial_offset = d1 * dim2 + d2;
    
    for (int f = 0; f < num_features; f++) {
        int in_idx = batch_offset + f * feature_stride + spatial_offset;
        float val = input[in_idx];
        sum_sq += val * val;
    }
    
    float inv_rms = rsqrtf(sum_sq / num_features + eps);
    
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
    
    // Use vectorized kernel for bulk of the work
    // dim2=512 is divisible by 4, so we can use vectorized kernel for all
    if (dim2 % 4 == 0) {
        const int total_vec4 = total_positions / 4;
        const int block_size = 256;
        const int num_blocks = (total_vec4 + block_size - 1) / block_size;
        
        rmsnorm_vectorized_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            num_features,
            dim1,
            dim2,
            eps
        );
    } else {
        const int block_size = 256;
        const int num_blocks = (total_positions + block_size - 1) / block_size;
        
        rmsnorm_scalar_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            num_features,
            dim1,
            dim2,
            eps,
            0
        );
    }
    
    return output;
}
"""

rmsnorm_cpp_source = """
torch::Tensor rmsnorm_hip(torch::Tensor input, float eps);
"""

rmsnorm_module = load_inline(
    name="rmsnorm_hip_v2",
    cpp_sources=rmsnorm_cpp_source,
    cuda_sources=rmsnorm_hip_source,
    functions=["rmsnorm_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized RMS Normalization using HIP kernel with vectorization.
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
