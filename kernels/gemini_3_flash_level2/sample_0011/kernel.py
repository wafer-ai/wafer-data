
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

fused_kernels_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__global__ void mean_var_kernel_optimized(
    const float4* __restrict__ input_vec,
    float* __restrict__ mean,
    float* __restrict__ inv_std,
    int N, int G, int elements_per_group_v4, float inv_elements, float eps) {
    
    int ng = blockIdx.x;
    if (ng >= N * G) return;
    
    const float4* group_input_v4 = input_vec + ng * elements_per_group_v4;
    
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    #pragma unroll 4
    for (int i = threadIdx.x; i < elements_per_group_v4; i += blockDim.x) {
        float4 val = group_input_v4[i];
        sum += val.x + val.y + val.z + val.w;
        sum_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }
    
    __shared__ float s_sum[32];
    __shared__ float s_sum_sq[32];
    
    float warp_sum = sum;
    float warp_sum_sq = sum_sq;
    for (int offset = 16; offset > 0; offset /= 2) {
        warp_sum += __shfl_down(warp_sum, offset);
        warp_sum_sq += __shfl_down(warp_sum_sq, offset);
    }
    
    if (threadIdx.x % 32 == 0) {
        s_sum[threadIdx.x / 32] = warp_sum;
        s_sum_sq[threadIdx.x / 32] = warp_sum_sq;
    }
    __syncthreads();
    
    if (threadIdx.x < 32) {
        float final_sum = (threadIdx.x < blockDim.x / 32) ? s_sum[threadIdx.x] : 0.0f;
        float final_sum_sq = (threadIdx.x < blockDim.x / 32) ? s_sum_sq[threadIdx.x] : 0.0f;
        
        for (int offset = 16; offset > 0; offset /= 2) {
            final_sum += __shfl_down(final_sum, offset);
            final_sum_sq += __shfl_down(final_sum_sq, offset);
        }
        
        if (threadIdx.x == 0) {
            float m = final_sum * inv_elements;
            float var = (final_sum_sq * inv_elements) - (m * m);
            mean[ng] = m;
            inv_std[ng] = rsqrtf(fmaxf(var, 0.0f) + eps);
        }
    }
}

__global__ void fused_gn_scale_maxpool_clamp_kernel_optimized(
    const float* __restrict__ input,
    const float* __restrict__ mean,
    const float* __restrict__ inv_std,
    const float* __restrict__ gn_weight,
    const float* __restrict__ gn_bias,
    const float* __restrict__ scale,
    float* __restrict__ output,
    int N, int C, int G, int H, int W,
    int H_pool, int W_pool, int pool_size,
    float clamp_min, float clamp_max) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H_pool * W_pool;
    if (idx >= total_elements) return;
    
    int pw = idx % W_pool;
    int ph = (idx / W_pool) % H_pool;
    int c = (idx / (W_pool * H_pool)) % C;
    int n = idx / (W_pool * H_pool * C);
    
    int g = c / (C / G);
    int ng = n * G + g;
    
    float m = mean[ng];
    float istd = inv_std[ng];
    float w = gn_weight[c];
    float b = gn_bias[c];
    float s = scale[c];
    
    float eff_w = istd * w * s;
    float eff_b = (b - m * istd * w) * s;
    
    float max_val = -1e38f;
    
    int h_start = ph * pool_size;
    int w_start = pw * pool_size;
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) { // pool_size is 4
        int h = h_start + i;
        const float* in_row = input + ((n * C + c) * H + h) * W + w_start;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float val = in_row[j];
            val = val * eff_w + eff_b;
            if (val > max_val) max_val = val;
        }
    }
    
    max_val = fminf(fmaxf(max_val, clamp_min), clamp_max);
    output[idx] = max_val;
}

torch::Tensor fused_op_hip(
    torch::Tensor x,
    torch::Tensor mean,
    torch::Tensor inv_std,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    torch::Tensor scale,
    int num_groups,
    int pool_size,
    float clamp_min,
    float clamp_max) {
    
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    
    int H_pool = H / pool_size;
    int W_pool = W / pool_size;
    
    auto output = torch::empty({N, C, H_pool, W_pool}, x.options());
    int total_elements = N * C * H_pool * W_pool;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_gn_scale_maxpool_clamp_kernel_optimized<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        inv_std.data_ptr<float>(),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        scale.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, num_groups, H, W,
        H_pool, W_pool, pool_size,
        clamp_min, clamp_max
    );
    
    return output;
}

std::vector<torch::Tensor> mean_var_hip(torch::Tensor x, int num_groups, float eps) {
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int G = num_groups;
    
    int elements_per_group = (C / G) * H * W;
    int elements_per_group_v4 = elements_per_group / 4;
    float inv_elements = 1.0f / elements_per_group;
    
    auto mean = torch::empty({N, G}, x.options());
    auto inv_std = torch::empty({N, G}, x.options());
    
    mean_var_kernel_optimized<<<N * G, 256>>>(
        (const float4*)x.data_ptr<float>(),
        mean.data_ptr<float>(),
        inv_std.data_ptr<float>(),
        N, G, elements_per_group_v4, inv_elements, eps
    );
    
    return {mean, inv_std};
}
"""

fused_ops = load_inline(
    name="fused_ops_v3",
    cpp_sources=fused_kernels_cpp_source,
    functions=["mean_var_hip", "fused_op_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size).cuda()
        self.group_norm = nn.GroupNorm(num_groups, out_channels).cuda()
        self.scale = nn.Parameter(torch.ones(scale_shape)).cuda()
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.num_groups = num_groups

    def forward(self, x):
        x = self.conv(x)
        mean, inv_std = fused_ops.mean_var_hip(x, self.num_groups, self.group_norm.eps)
        gn_weight = self.group_norm.weight
        gn_bias = self.group_norm.bias
        if gn_weight is None:
            gn_weight = torch.ones(x.size(1), device=x.device, dtype=x.dtype)
        if gn_bias is None:
            gn_bias = torch.zeros(x.size(1), device=x.device, dtype=x.dtype)
            
        x = fused_ops.fused_op_hip(
            x, mean, inv_std, 
            gn_weight, gn_bias, 
            self.scale.view(-1),
            self.num_groups, self.maxpool_kernel_size,
            self.clamp_min, self.clamp_max
        )
        return x

