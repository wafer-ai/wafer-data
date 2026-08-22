import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused InstanceNorm + Division kernel optimized for MI300X with single-pass Welford
fused_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define WARP_SIZE 64

__device__ __forceinline__ void warpReduceSumTwo(float& v1, float& v2) {
    #pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        v1 += __shfl_down(v1, offset);
        v2 += __shfl_down(v2, offset);
    }
}

// Kernel optimized for 126x126 spatial size (after 3x3 conv on 128x128)
// HW = 15876, which divides by 4 = 3969
__global__ void fused_instnorm_div_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int C, int HW,
    float inv_divide_eps_scale,  // rsqrt(eps) / divide_by precomputed
    float eps
) {
    __shared__ float s_data[8];  // For sum and sum_sq from 4 warps
    __shared__ float s_mean_inv_std[2];  // [mean, inv_std_div]
    
    const int bc_idx = blockIdx.x;
    const int offset = bc_idx * HW;
    
    const float* in_ptr = input + offset;
    float* out_ptr = output + offset;
    
    // First pass: compute sum and sum of squares using vectorized loads
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    const int vec_HW = HW >> 2;  // HW / 4
    const float4* in_vec = reinterpret_cast<const float4*>(in_ptr);
    
    for (int i = threadIdx.x; i < vec_HW; i += blockDim.x) {
        float4 v = in_vec[i];
        float s = v.x + v.y + v.z + v.w;
        float sq = v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
        local_sum += s;
        local_sum_sq += sq;
    }
    
    // Handle remaining elements (if HW not divisible by 4)
    int remainder_start = vec_HW << 2;
    for (int i = remainder_start + threadIdx.x; i < HW; i += blockDim.x) {
        float val = in_ptr[i];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    // Warp reduction
    warpReduceSumTwo(local_sum, local_sum_sq);
    
    int warp_id = threadIdx.x >> 6;  // / 64
    int lane = threadIdx.x & 63;     // % 64
    
    if (lane == 0) {
        s_data[warp_id] = local_sum;
        s_data[warp_id + 4] = local_sum_sq;
    }
    __syncthreads();
    
    // Final reduction in first 4 threads
    if (threadIdx.x < 4) {
        local_sum = s_data[threadIdx.x];
        local_sum_sq = s_data[threadIdx.x + 4];
        
        // Reduction among 4 threads
        local_sum += __shfl_down(local_sum, 2);
        local_sum_sq += __shfl_down(local_sum_sq, 2);
        local_sum += __shfl_down(local_sum, 1);
        local_sum_sq += __shfl_down(local_sum_sq, 1);
        
        if (threadIdx.x == 0) {
            float inv_HW = 1.0f / (float)HW;
            float mean = local_sum * inv_HW;
            float var = local_sum_sq * inv_HW - mean * mean;
            s_mean_inv_std[0] = mean;
            s_mean_inv_std[1] = rsqrtf(var + eps) * inv_divide_eps_scale;
        }
    }
    __syncthreads();
    
    float mean = s_mean_inv_std[0];
    float inv_std_div = s_mean_inv_std[1];
    
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
    for (int i = remainder_start + threadIdx.x; i < HW; i += blockDim.x) {
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
    
    float inv_divide_eps_scale = 1.0f / divide_by;
    
    fused_instnorm_div_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        C, HW,
        inv_divide_eps_scale,
        eps
    );
    
    return output;
}
"""

fused_header = """
torch::Tensor fused_instnorm_div_hip(torch::Tensor input, float divide_by, float eps);
"""

fused_mod = load_inline(
    name="fused_instnorm_div_v6",
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
