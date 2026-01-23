
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

group_norm_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

__device__ __forceinline__ void warpReduceSumSq(float& sum, float& sum_sq) {
    for (int offset = 64 / 2; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset, 64);
        sum_sq += __shfl_down(sum_sq, offset, 64);
    }
}

__global__ void group_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int N, int C, int G, int HW, int C_per_G, float eps) {

    int n = blockIdx.x;
    int g = blockIdx.y;
    int M = C_per_G * HW;
    int base_idx = (n * C + g * C_per_G) * HW;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    const float4* x_ptr4 = reinterpret_cast<const float4*>(x + base_idx);
    int M4 = M >> 2;
    for (int k = threadIdx.x; k < M4; k += blockDim.x) {
        float4 val4 = x_ptr4[k];
        sum += val4.x + val4.y + val4.z + val4.w;
        sum_sq += val4.x * val4.x + val4.y * val4.y + val4.z * val4.z + val4.w * val4.w;
    }

    __shared__ float s_sum[4];
    __shared__ float s_sum_sq[4];
    
    int lane = threadIdx.x & 63;
    int wid = threadIdx.x >> 6;

    warpReduceSumSq(sum, sum_sq);

    if (lane == 0) {
        s_sum[wid] = sum;
        s_sum_sq[wid] = sum_sq;
    }
    __syncthreads();

    if (wid == 0) {
        sum = (threadIdx.x < (blockDim.x >> 6)) ? s_sum[lane] : 0.0f;
        sum_sq = (threadIdx.x < (blockDim.x >> 6)) ? s_sum_sq[lane] : 0.0f;
        warpReduceSumSq(sum, sum_sq);
        if (lane == 0) {
            s_sum[0] = sum;
            s_sum_sq[0] = sum_sq;
        }
    }
    __syncthreads();

    float mean = s_sum[0] / M;
    float var = fmaxf(s_sum_sq[0] / M - mean * mean, 0.0f);
    float inv_std = 1.0f / sqrtf(var + eps);

    for (int i = 0; i < C_per_G; ++i) {
        int c = g * C_per_G + i;
        float g_val = gamma[c];
        float b_val = beta[c];
        
        const float4* x_c_ptr4 = reinterpret_cast<const float4*>(x + (n * C + c) * HW);
        float4* y_c_ptr4 = reinterpret_cast<float4*>(y + (n * C + c) * HW);
        int HW4 = HW >> 2;
        
        for (int j = threadIdx.x; j < HW4; j += blockDim.x) {
            float4 val4 = x_c_ptr4[j];
            float4 res4;
            res4.x = fmaf(g_val, (val4.x - mean) * inv_std, b_val);
            res4.y = fmaf(g_val, (val4.y - mean) * inv_std, b_val);
            res4.z = fmaf(g_val, (val4.z - mean) * inv_std, b_val);
            res4.w = fmaf(g_val, (val4.w - mean) * inv_std, b_val);
            y_c_ptr4[j] = res4;
        }
    }
}

torch::Tensor group_norm_hip(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, int num_groups, float eps) {
    int N = x.size(0);
    int C = x.size(1);
    int G = num_groups;
    int HW = 1;
    for (int i = 2; i < x.dim(); ++i) {
        HW *= x.size(i);
    }
    int C_per_G = C / G;

    auto y = torch::empty_like(x);

    dim3 grid(N, G);
    dim3 block(256);

    hipLaunchKernelGGL(group_norm_kernel, grid, block, 0, 0,
        x.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), y.data_ptr<float>(),
        N, C, G, HW, C_per_G, eps);

    return y;
}
"""

group_norm_cpp_source = """
torch::Tensor group_norm_hip(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, int num_groups, float eps);
"""

group_norm_module = load_inline(
    name="group_norm_module",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return group_norm_module.group_norm_hip(x, self.weight, self.bias, self.num_groups, self.eps)
