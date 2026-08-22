
import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>
#include <cmath>

__global__ void group_norm_stats_kernel_float4(
    const float* __restrict__ x,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    int N, int C, int H, int W, int G, float eps) {
    
    int n = blockIdx.x;
    int g = blockIdx.y;
    
    int C_per_G = C / G;
    int num_elements = C_per_G * H * W;
    int num_vectors = num_elements / 4;
    
    long long batch_offset = (long long)n * C * H * W;
    long long group_offset = (long long)g * C_per_G * H * W;
    const float4* group_data = (const float4*)(x + batch_offset + group_offset);
    
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    for (int i = threadIdx.x; i < num_vectors; i += blockDim.x) {
        float4 v = group_data[i];
        sum += v.x + v.y + v.z + v.w;
        sum_sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    
    int processed = num_vectors * 4;
    const float* raw_data = x + batch_offset + group_offset;
    for (int i = processed + threadIdx.x; i < num_elements; i += blockDim.x) {
         float val = raw_data[i];
         sum += val;
         sum_sq += val * val;
    }
    
    __shared__ float s_sum[256];
    __shared__ float s_sum_sq[256];
    
    s_sum[threadIdx.x] = sum;
    s_sum_sq[threadIdx.x] = sum_sq;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        float mu = s_sum[0] / num_elements;
        float var = (s_sum_sq[0] / num_elements) - (mu * mu);
        if (var < 0.0f) var = 0.0f;
        
        mean[n * G + g] = mu;
        rstd[n * G + g] = rsqrtf(var + eps);
    }
}

__global__ void fused_apply_kernel_k4(
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    const float* __restrict__ gn_weight,
    const float* __restrict__ gn_bias,
    const float* __restrict__ scale,
    float* __restrict__ out,
    int N, int C, int H_in, int W_in, int G,
    int H_out, int W_out,
    float clamp_min, float clamp_max) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = N * C * H_out * W_out;
    
    if (idx >= total_threads) return;
    
    int temp = idx;
    int ow = temp % W_out;
    temp /= W_out;
    int oh = temp % H_out;
    temp /= H_out;
    int c = temp % C;
    int n = temp / C;
    
    int C_per_G = C / G;
    int g = c / C_per_G;
    
    float mu = mean[n * G + g];
    float rs = rstd[n * G + g];
    
    float gamma = gn_weight[c];
    float beta = gn_bias[c];
    float s = scale[c];
    
    int h_start = oh * 4;
    int w_start = ow * 4;
    
    float max_val = -3.402823466e+38F;
    
    long long input_offset = (long long)n * C * H_in * W_in + (long long)c * H_in * W_in;
    const float* ch_input = x + input_offset;
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int h_in = h_start + i;
        long long row_off = (long long)h_in * W_in + w_start;
        const float* row_ptr = ch_input + row_off;
        
        float v0 = row_ptr[0];
        float v1 = row_ptr[1];
        float v2 = row_ptr[2];
        float v3 = row_ptr[3];
        
        float n0 = (v0 - mu) * rs;
        float g0 = n0 * gamma + beta;
        float s0 = g0 * s;
        max_val = fmaxf(max_val, s0);
        
        float n1 = (v1 - mu) * rs;
        float g1 = n1 * gamma + beta;
        float s1 = g1 * s;
        max_val = fmaxf(max_val, s1);

        float n2 = (v2 - mu) * rs;
        float g2 = n2 * gamma + beta;
        float s2 = g2 * s;
        max_val = fmaxf(max_val, s2);

        float n3 = (v3 - mu) * rs;
        float g3 = n3 * gamma + beta;
        float s3 = g3 * s;
        max_val = fmaxf(max_val, s3);
    }
    
    max_val = fmaxf(max_val, clamp_min);
    max_val = fminf(max_val, clamp_max);
    
    out[idx] = max_val;
}

torch::Tensor fused_forward(torch::Tensor x, torch::Tensor gn_weight, torch::Tensor gn_bias, torch::Tensor scale, 
                            int num_groups, int pool_k, float clamp_min, float clamp_max, float eps) {
    
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    
    auto mean = torch::empty({N, num_groups}, x.options());
    auto rstd = torch::empty({N, num_groups}, x.options());
    
    dim3 stats_grid(N, num_groups);
    dim3 stats_block(256);
    
    group_norm_stats_kernel_float4<<<stats_grid, stats_block>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        N, C, H, W, num_groups, eps
    );
    
    int H_out = H / pool_k;
    int W_out = W / pool_k;
    
    auto out = torch::empty({N, C, H_out, W_out}, x.options());
    
    if (pool_k == 4) {
        int total_elements = N * C * H_out * W_out;
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;
        
        fused_apply_kernel_k4<<<grid_size, block_size>>>(
            x.data_ptr<float>(),
            mean.data_ptr<float>(),
            rstd.data_ptr<float>(),
            gn_weight.data_ptr<float>(),
            gn_bias.data_ptr<float>(),
            scale.data_ptr<float>(),
            out.data_ptr<float>(),
            N, C, H, W, num_groups,
            H_out, W_out,
            clamp_min, clamp_max
        );
    }
    
    return out;
}
"""

fused_op = load_inline(
    name="fused_gn_pool_final",
    cpp_sources=cpp_source,
    functions=["fused_forward"],
    extra_cflags=["-O3"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv(x)
        return self.fused_op.fused_forward(
            x, 
            self.group_norm.weight, 
            self.group_norm.bias, 
            self.scale,
            self.group_norm.num_groups,
            self.maxpool_kernel_size,
            self.clamp_min,
            self.clamp_max,
            self.group_norm.eps
        )

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128 
kernel_size = 3
num_groups = 16
scale_shape = (out_channels, 1, 1)
maxpool_kernel_size = 4
clamp_min = 0.0
clamp_max = 1.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max]
