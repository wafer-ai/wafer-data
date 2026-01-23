
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

instance_norm_divide_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    for (int offset = 32; offset > 0; offset /= 2)
        val += __shfl_xor(val, offset);
    return val;
}

__global__ void instance_norm_divide_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    float eps,
    float divide_by) {

    int nc = blockIdx.x;
    if (nc >= N * C) return;

    int offset = nc * H * W;
    int size = H * W;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    // Use float4 for vectorized loads
    const float4* input4 = reinterpret_cast<const float4*>(input + offset);
    int size4 = size / 4;
    
    for (int i = threadIdx.x; i < size4; i += blockDim.x) {
        float4 val4 = input4[i];
        local_sum += val4.x + val4.y + val4.z + val4.w;
        local_sum_sq += val4.x * val4.x + val4.y * val4.y + val4.z * val4.z + val4.w * val4.w;
    }

    // Handle remainder
    for (int i = size4 * 4 + threadIdx.x; i < size; i += blockDim.x) {
        float val = input[offset + i];
        local_sum += val;
        local_sum_sq += val * val;
    }

    // Block-level reduction using shared memory
    extern __shared__ float s_mem[];
    float* s_sum = s_mem;
    float* s_sum_sq = s_mem + blockDim.x;

    s_sum[threadIdx.x] = local_sum;
    s_sum_sq[threadIdx.x] = local_sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / size;
    float var = (s_sum_sq[0] / size) - (mean * mean);
    if (var < 0.0f) var = 0.0f;
    float inv_std = 1.0f / (sqrtf(var + eps) * divide_by);

    float4* output4 = reinterpret_cast<float4*>(output + offset);
    for (int i = threadIdx.x; i < size4; i += blockDim.x) {
        float4 val4 = input4[i];
        float4 res4;
        res4.x = (val4.x - mean) * inv_std;
        res4.y = (val4.y - mean) * inv_std;
        res4.z = (val4.z - mean) * inv_std;
        res4.w = (val4.w - mean) * inv_std;
        output4[i] = res4;
    }

    for (int i = size4 * 4 + threadIdx.x; i < size; i += blockDim.x) {
        output[offset + i] = (input[offset + i] - mean) * inv_std;
    }
}

torch::Tensor instance_norm_divide_hip(torch::Tensor input, float eps, float divide_by) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);

    auto output = torch::empty_like(input);
    int threads = 256;
    int blocks = N * C;
    int shared_size = 2 * threads * sizeof(float);

    instance_norm_divide_kernel<<<blocks, threads, shared_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        eps,
        divide_by
    );

    return output;
}
"""

instance_norm_divide_module = load_inline(
    name="instance_norm_divide",
    cpp_sources=instance_norm_divide_source,
    functions=["instance_norm_divide_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size).cuda()
        self.divide_by = float(divide_by)
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv(x)
        x = instance_norm_divide_module.instance_norm_divide_hip(x, self.eps, self.divide_by)
        return x
