import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused Instance Normalization + Divide kernel - optimized version
instance_norm_divide_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

#define WARP_SIZE 64

// Helper: warp reduce sum
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset, WARP_SIZE);
    }
    return val;
}

// Helper: block reduce sum
__device__ float blockReduceSum(float val, float* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warpReduceSum(val);
    
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
    
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    
    return val;
}

// Optimized kernel with vectorized loads
__global__ void instance_norm_divide_kernel_opt(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int spatial_size,
    float inv_divide_by,
    float eps
) {
    // Each block processes one (n, c) pair
    int nc = blockIdx.x;
    int n = nc / channels;
    int c = nc % channels;
    
    if (n >= batch_size) return;
    
    const float* in_ptr = input + (n * channels + c) * spatial_size;
    float* out_ptr = output + (n * channels + c) * spatial_size;
    
    extern __shared__ float shared_mem[];
    
    // Vector loads for better memory bandwidth (float4)
    int vec_size = spatial_size / 4;
    const float4* in_ptr4 = reinterpret_cast<const float4*>(in_ptr);
    
    // Compute mean using parallel reduction with vectorized loads
    float sum = 0.0f;
    
    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        float4 val = in_ptr4[i];
        sum += val.x + val.y + val.z + val.w;
    }
    
    // Handle remainder
    for (int i = vec_size * 4 + threadIdx.x; i < spatial_size; i += blockDim.x) {
        sum += in_ptr[i];
    }
    
    sum = blockReduceSum(sum, shared_mem);
    
    __shared__ float mean_shared;
    if (threadIdx.x == 0) {
        mean_shared = sum / spatial_size;
    }
    __syncthreads();
    float mean = mean_shared;
    
    // Compute variance with vectorized loads
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        float4 val = in_ptr4[i];
        float d0 = val.x - mean;
        float d1 = val.y - mean;
        float d2 = val.z - mean;
        float d3 = val.w - mean;
        var_sum += d0*d0 + d1*d1 + d2*d2 + d3*d3;
    }
    
    for (int i = vec_size * 4 + threadIdx.x; i < spatial_size; i += blockDim.x) {
        float diff = in_ptr[i] - mean;
        var_sum += diff * diff;
    }
    
    var_sum = blockReduceSum(var_sum, shared_mem);

    __shared__ float inv_std_shared;
    if (threadIdx.x == 0) {
        float var = var_sum / spatial_size;
        inv_std_shared = rsqrtf(var + eps) * inv_divide_by;
    }
    __syncthreads();
    float inv_std_div = inv_std_shared;
    
    // Normalize and divide with vectorized stores
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
    int shared_mem_size = (threads_per_block / WARP_SIZE) * sizeof(float);
    
    float inv_divide_by = 1.0f / divide_by;
    
    instance_norm_divide_kernel_opt<<<num_blocks, threads_per_block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        spatial_size,
        inv_divide_by,
        eps
    );
    
    return output;
}
"""

instance_norm_divide_cpp = """
torch::Tensor instance_norm_divide_hip(torch::Tensor input, float divide_by, float eps);
"""

instance_norm_divide = load_inline(
    name="instance_norm_divide_v5",
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
