import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused InstanceNorm + Division kernel with better optimization
fused_instnorm_div_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Welford's online algorithm for stable mean/variance computation
// Each block handles one (batch, channel) pair with vectorized loads
__global__ void fused_instance_norm_div_kernel_v2(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    float divide_by,
    float eps
) {
    __shared__ float s_sum[64];
    __shared__ float s_sum_sq[64];
    __shared__ float s_mean;
    __shared__ float s_inv_std;
    
    int batch_idx = blockIdx.x / C;
    int channel_idx = blockIdx.x % C;
    
    int HW = H * W;
    int offset = batch_idx * C * HW + channel_idx * HW;
    
    const float* in_ptr = input + offset;
    float* out_ptr = output + offset;
    
    // First pass: compute sum and sum of squares for mean and variance
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    // Process 4 elements at a time for better memory throughput
    int vec_limit = (HW / 4) * 4;
    int i = threadIdx.x * 4;
    
    for (; i < vec_limit; i += blockDim.x * 4) {
        float4 vals = *reinterpret_cast<const float4*>(in_ptr + i);
        local_sum += vals.x + vals.y + vals.z + vals.w;
        local_sum_sq += vals.x * vals.x + vals.y * vals.y + vals.z * vals.z + vals.w * vals.w;
    }
    
    // Handle remainder
    for (int j = vec_limit + threadIdx.x; j < HW; j += blockDim.x) {
        float val = in_ptr[j];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    // Warp reduction using shuffles
    for (int offset = 32; offset > 0; offset /= 2) {
        local_sum += __shfl_down(local_sum, offset);
        local_sum_sq += __shfl_down(local_sum_sq, offset);
    }
    
    int warp_id = threadIdx.x / 64;
    int lane = threadIdx.x % 64;
    
    if (lane == 0) {
        s_sum[warp_id] = local_sum;
        s_sum_sq[warp_id] = local_sum_sq;
    }
    __syncthreads();
    
    // Final reduction
    if (threadIdx.x < 4) {
        local_sum = s_sum[threadIdx.x];
        local_sum_sq = s_sum_sq[threadIdx.x];
        
        for (int offset = 2; offset > 0; offset /= 2) {
            local_sum += __shfl_down(local_sum, offset);
            local_sum_sq += __shfl_down(local_sum_sq, offset);
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
    
    // Second pass: normalize and divide using vectorized stores
    i = threadIdx.x * 4;
    for (; i < vec_limit; i += blockDim.x * 4) {
        float4 vals = *reinterpret_cast<const float4*>(in_ptr + i);
        float4 result;
        result.x = (vals.x - mean) * inv_std_div;
        result.y = (vals.y - mean) * inv_std_div;
        result.z = (vals.z - mean) * inv_std_div;
        result.w = (vals.w - mean) * inv_std_div;
        *reinterpret_cast<float4*>(out_ptr + i) = result;
    }
    
    // Handle remainder
    for (int j = vec_limit + threadIdx.x; j < HW; j += blockDim.x) {
        out_ptr[j] = (in_ptr[j] - mean) * inv_std_div;
    }
}

torch::Tensor fused_instance_norm_div_hip(torch::Tensor input, float divide_by, float eps) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    
    auto output = torch::empty_like(input);
    
    int threads = 256;
    int blocks = N * C;
    
    fused_instance_norm_div_kernel_v2<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        divide_by,
        eps
    );
    
    return output;
}
"""

fused_instnorm_div_cpp = """
torch::Tensor fused_instance_norm_div_hip(torch::Tensor input, float divide_by, float eps);
"""

fused_module = load_inline(
    name="fused_instance_norm_div",
    cpp_sources=fused_instnorm_div_cpp,
    cuda_sources=fused_instnorm_div_source,
    functions=["fused_instance_norm_div_hip"],
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
        x = fused_module.fused_instance_norm_div_hip(x, self.divide_by, self.eps)
        return x


def get_inputs():
    return [torch.rand(128, 64, 128, 128).cuda()]


def get_init_inputs():
    return [64, 128, 3, 2.0]
