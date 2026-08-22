import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused InstanceNorm + Division kernel optimized for MI300X
fused_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define WARP_SIZE 64

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 32; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Kernel: 256 threads, vectorized loads
__global__ void fused_instnorm_div_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int C, int HW,
    float divide_by,
    float eps
) {
    __shared__ float s_sum[4];
    __shared__ float s_sum_sq[4];
    __shared__ float s_mean;
    __shared__ float s_inv_std;
    
    const int bc_idx = blockIdx.x;
    const int batch_idx = bc_idx / C;
    const int channel_idx = bc_idx % C;
    const int offset = batch_idx * C * HW + channel_idx * HW;
    
    const float* in_ptr = input + offset;
    float* out_ptr = output + offset;
    
    // First pass: compute sum and sum of squares
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    // Vectorized loads - process 4 elements at once
    const int vec_HW = HW / 4;
    const float4* in_vec = reinterpret_cast<const float4*>(in_ptr);
    
    for (int i = threadIdx.x; i < vec_HW; i += blockDim.x) {
        float4 v = in_vec[i];
        local_sum += v.x + v.y + v.z + v.w;
        local_sum_sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    
    // Handle remaining elements
    for (int i = vec_HW * 4 + threadIdx.x; i < HW; i += blockDim.x) {
        float val = in_ptr[i];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    // Warp reduction
    local_sum = warpReduceSum(local_sum);
    local_sum_sq = warpReduceSum(local_sum_sq);
    
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    
    if (lane == 0) {
        s_sum[warp_id] = local_sum;
        s_sum_sq[warp_id] = local_sum_sq;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (threadIdx.x < 4) {
        local_sum = s_sum[threadIdx.x];
        local_sum_sq = s_sum_sq[threadIdx.x];
        
        for (int off = 2; off > 0; off /= 2) {
            local_sum += __shfl_down(local_sum, off);
            local_sum_sq += __shfl_down(local_sum_sq, off);
        }
        
        if (threadIdx.x == 0) {
            float mean = local_sum / (float)HW;
            float var = local_sum_sq / (float)HW - mean * mean;
            s_mean = mean;
            s_inv_std = rsqrtf(var + eps) / divide_by;
        }
    }
    __syncthreads();
    
    float mean = s_mean;
    float inv_std_div = s_inv_std;
    
    // Second pass: normalize with vectorized stores
    float4* out_vec = reinterpret_cast<float4*>(out_ptr);
    
    for (int i = threadIdx.x; i < vec_HW; i += blockDim.x) {
        float4 v = in_vec[i];
        float4 result;
        result.x = (v.x - mean) * inv_std_div;
        result.y = (v.y - mean) * inv_std_div;
        result.z = (v.z - mean) * inv_std_div;
        result.w = (v.w - mean) * inv_std_div;
        out_vec[i] = result;
    }
    
    // Handle remaining elements
    for (int i = vec_HW * 4 + threadIdx.x; i < HW; i += blockDim.x) {
        out_ptr[i] = (in_ptr[i] - mean) * inv_std_div;
    }
}

torch::Tensor fused_instnorm_div_hip(torch::Tensor input, float divide_by, float eps) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    int HW = H * W;
    
    auto output = torch::empty_like(input);
    
    int threads = 256;
    int blocks = N * C;
    
    fused_instnorm_div_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        C, HW,
        divide_by,
        eps
    );
    
    return output;
}
"""

fused_header = """
torch::Tensor fused_instnorm_div_hip(torch::Tensor input, float divide_by, float eps);
"""

fused_mod = load_inline(
    name="fused_instnorm_div_v5",
    cpp_sources=fused_header,
    cuda_sources=fused_source,
    functions=["fused_instnorm_div_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    """
    Optimized model with fused InstanceNorm + Division kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divide_by = divide_by
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv(x)
        x = fused_mod.fused_instnorm_div_hip(x, self.divide_by, self.eps)
        return x


def get_inputs():
    return [torch.rand(128, 64, 128, 128).cuda()]


def get_init_inputs():
    return [64, 128, 3, 2.0]
