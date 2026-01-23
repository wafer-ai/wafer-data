import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused Instance Normalization + Divide kernel - Welford's algorithm
instance_norm_divide_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

#define WARP_SIZE 64

// Welford's online algorithm for mean and variance in one pass
__global__ void instance_norm_divide_welford_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int spatial_size,
    float divide_by,
    float eps
) {
    // Each block processes one (n, c) pair
    int nc = blockIdx.x;
    int n = nc / channels;
    int c = nc % channels;
    
    if (n >= batch_size) return;
    
    const float* in_ptr = input + (n * channels + c) * spatial_size;
    float* out_ptr = output + (n * channels + c) * spatial_size;
    
    __shared__ float shared_sum[16];
    __shared__ float shared_sum_sq[16];
    
    // Phase 1: Compute sum and sum of squares in parallel
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    // Vectorized loads (float4)
    int vec_size = spatial_size / 4;
    const float4* in_ptr4 = reinterpret_cast<const float4*>(in_ptr);
    
    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        float4 val = in_ptr4[i];
        local_sum += val.x + val.y + val.z + val.w;
        local_sum_sq += val.x*val.x + val.y*val.y + val.z*val.z + val.w*val.w;
    }
    
    // Handle remainder
    for (int i = vec_size * 4 + threadIdx.x; i < spatial_size; i += blockDim.x) {
        float val = in_ptr[i];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    // Warp reduce
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down(local_sum, offset, WARP_SIZE);
        local_sum_sq += __shfl_down(local_sum_sq, offset, WARP_SIZE);
    }
    
    if (lane == 0) {
        shared_sum[wid] = local_sum;
        shared_sum_sq[wid] = local_sum_sq;
    }
    __syncthreads();
    
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    local_sum = (threadIdx.x < num_warps) ? shared_sum[threadIdx.x] : 0.0f;
    local_sum_sq = (threadIdx.x < num_warps) ? shared_sum_sq[threadIdx.x] : 0.0f;
    
    if (wid == 0) {
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            local_sum += __shfl_down(local_sum, offset, WARP_SIZE);
            local_sum_sq += __shfl_down(local_sum_sq, offset, WARP_SIZE);
        }
    }
    
    __shared__ float mean_shared;
    __shared__ float inv_std_div_shared;
    
    if (threadIdx.x == 0) {
        float mean = local_sum / spatial_size;
        // Var = E[X^2] - E[X]^2
        float var = local_sum_sq / spatial_size - mean * mean;
        mean_shared = mean;
        inv_std_div_shared = rsqrtf(var + eps) / divide_by;
    }
    __syncthreads();
    
    float mean = mean_shared;
    float inv_std_div = inv_std_div_shared;
    
    // Phase 2: Normalize and divide with vectorized stores
    float4* out_ptr4 = reinterpret_cast<float4*>(out_ptr);
    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        float4 val = in_ptr4[i];
        float4 result;
        result.x = (val.x - mean) * inv_std_div;
        result.y = (val.y - mean) * inv_std_div;
        result.z = (val.z - mean) * inv_std_div;
        result.w = (val.w - mean) * inv_std_div;
        out_ptr4[i] = result;
    }
    
    for (int i = vec_size * 4 + threadIdx.x; i < spatial_size; i += blockDim.x) {
        out_ptr[i] = (in_ptr[i] - mean) * inv_std_div;
    }
}

torch::Tensor instance_norm_divide_hip(torch::Tensor input, float divide_by, float eps) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto spatial_size = height * width;
    
    auto output = torch::empty_like(input);
    
    int num_blocks = batch_size * channels;
    int threads_per_block = 512;
    
    instance_norm_divide_welford_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        spatial_size,
        divide_by,
        eps
    );
    
    return output;
}
"""

instance_norm_divide_cpp = """
torch::Tensor instance_norm_divide_hip(torch::Tensor input, float divide_by, float eps);
"""

instance_norm_divide = load_inline(
    name="instance_norm_divide_v3",
    cpp_sources=instance_norm_divide_cpp,
    cuda_sources=instance_norm_divide_source,
    functions=["instance_norm_divide_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, applies fused Instance Normalization + Division.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divide_by = divide_by
        self.eps = 1e-5
        self.instance_norm_divide = instance_norm_divide

    def forward(self, x):
        x = self.conv(x)
        x = self.instance_norm_divide.instance_norm_divide_hip(x, self.divide_by, self.eps)
        return x
