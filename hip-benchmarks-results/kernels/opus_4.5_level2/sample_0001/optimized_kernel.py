import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused InstanceNorm + Division kernel
fused_instnorm_div_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 32; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val, float* shared) {
    int lane = threadIdx.x % 64;
    int wid = threadIdx.x / 64;
    
    // Warp-level reduction
    for (int offset = 32; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    
    // Final reduction across warps
    int numWarps = (blockDim.x + 63) / 64;
    val = (threadIdx.x < numWarps) ? shared[threadIdx.x] : 0.0f;
    
    if (wid == 0) {
        for (int offset = 32; offset > 0; offset /= 2) {
            val += __shfl_down(val, offset);
        }
    }
    
    return val;
}

// Each block handles one (batch, channel) pair
__global__ void fused_instance_norm_div_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    float divide_by,
    float eps
) {
    __shared__ float s_data[16]; // For reductions
    __shared__ float s_mean;
    __shared__ float s_inv_std;
    
    int batch_idx = blockIdx.x / C;
    int channel_idx = blockIdx.x % C;
    
    int HW = H * W;
    int offset = batch_idx * C * HW + channel_idx * HW;
    
    const float* in_ptr = input + offset;
    float* out_ptr = output + offset;
    
    // Compute sum for mean
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        local_sum += in_ptr[i];
    }
    
    local_sum = blockReduceSum(local_sum, s_data);
    
    if (threadIdx.x == 0) {
        s_mean = local_sum / (float)HW;
    }
    __syncthreads();
    
    float mean = s_mean;
    
    // Compute variance
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        float diff = in_ptr[i] - mean;
        local_var += diff * diff;
    }
    
    local_var = blockReduceSum(local_var, s_data);
    
    if (threadIdx.x == 0) {
        s_inv_std = rsqrtf(local_var / (float)HW + eps) / divide_by;
    }
    __syncthreads();
    
    float inv_std_div = s_inv_std;
    
    // Normalize and divide
    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        out_ptr[i] = (in_ptr[i] - mean) * inv_std_div;
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
    
    fused_instance_norm_div_kernel<<<blocks, threads>>>(
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
