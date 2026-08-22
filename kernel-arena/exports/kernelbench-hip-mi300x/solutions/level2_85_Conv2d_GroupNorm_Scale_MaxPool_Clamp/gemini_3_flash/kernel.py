
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

# HIP kernel source
hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

__global__ void compute_mean_var_kernel(
    const float* __restrict__ input,
    float* __restrict__ mean,
    float* __restrict__ inv_std,
    int batch_size,
    int num_groups,
    int channels_per_group,
    int height,
    int width,
    float eps) {

    int b = blockIdx.y;
    int g = blockIdx.x;

    if (b >= batch_size || g >= num_groups) return;

    int num_elements = channels_per_group * height * width;
    double sum = 0.0;
    double sum_sq = 0.0;

    int group_offset = (b * num_groups + g) * num_elements;

    for (int i = threadIdx.x; i < num_elements; i += blockDim.x) {
        float val = input[group_offset + i];
        sum += (double)val;
        sum_sq += (double)val * val;
    }

    // Using block reduction
    extern __shared__ double shared_mem[];
    double* s_sum = shared_mem;
    double* s_sum_sq = &shared_mem[blockDim.x];

    s_sum[threadIdx.x] = sum;
    s_sum_sq[threadIdx.x] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        double m = s_sum[0] / num_elements;
        double v = (s_sum_sq[0] / num_elements) - (m * m);
        if (v < 0) v = 0;
        mean[b * num_groups + g] = (float)m;
        inv_std[b * num_groups + g] = (float)(1.0 / sqrt(v + (double)eps));
    }
}

__global__ void fused_gn_scale_maxpool_clamp_kernel(
    const float* __restrict__ input,
    const float* __restrict__ mean,
    const float* __restrict__ inv_std,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ extra_scale,
    float* __restrict__ output,
    int batch_size,
    int num_groups,
    int channels_per_group,
    int input_h,
    int input_w,
    int output_h,
    int output_w,
    int pool_k,
    float clamp_min,
    float clamp_max) {

    int b = blockIdx.z;
    int c = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = idx / output_w;
    int ow = idx % output_w;

    if (b >= batch_size || c >= (num_groups * channels_per_group) || oh >= output_h) return;

    int g = c / channels_per_group;
    float m = mean[b * num_groups + g];
    float is = inv_std[b * num_groups + g];
    float w = weight[c];
    float bi = bias[c];
    float s = extra_scale[c];

    float fused_w = w * is * s;
    float fused_b = (bi - m * w * is) * s;

    float max_val = -1e38f;

    int ih_start = oh * pool_k;
    int iw_start = ow * pool_k;

    int channel_offset = (b * (num_groups * channels_per_group) + c) * input_h * input_w;

    for (int kh = 0; kh < pool_k; ++kh) {
        int ih = ih_start + kh;
        if (ih < input_h) {
            int row_offset = channel_offset + ih * input_w;
            for (int kw = 0; kw < pool_k; ++kw) {
                int iw = iw_start + kw;
                if (iw < input_w) {
                    float val = input[row_offset + iw];
                    float norm_val = val * fused_w + fused_b;
                    if (norm_val > max_val) {
                        max_val = norm_val;
                    }
                }
            }
        }
    }

    if (max_val < clamp_min) max_val = clamp_min;
    if (max_val > clamp_max) max_val = clamp_max;

    output[((b * (num_groups * channels_per_group) + c) * output_h + oh) * output_w + ow] = max_val;
}

torch::Tensor fused_op(
    torch::Tensor input,
    torch::Tensor mean_tensor,
    torch::Tensor inv_std_tensor,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor extra_scale,
    int output_h,
    int output_w,
    int pool_k,
    float clamp_min,
    float clamp_max) {

    int batch_size = input.size(0);
    int num_channels = input.size(1);
    int input_h = input.size(2);
    int input_w = input.size(3);
    int num_groups = mean_tensor.size(1);
    int channels_per_group = num_channels / num_groups;

    auto output = torch::empty({batch_size, num_channels, output_h, output_w}, input.options());

    dim3 block(256);
    dim3 grid((output_h * output_w + 255) / 256, num_channels, batch_size);

    fused_gn_scale_maxpool_clamp_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        mean_tensor.data_ptr<float>(),
        inv_std_tensor.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        extra_scale.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_groups,
        channels_per_group,
        input_h,
        input_w,
        output_h,
        output_w,
        pool_k,
        clamp_min,
        clamp_max
    );

    return output;
}

std::vector<torch::Tensor> compute_mean_var(torch::Tensor input, int num_groups, float eps) {
    int batch_size = input.size(0);
    int num_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int channels_per_group = num_channels / num_groups;

    auto mean = torch::empty({batch_size, num_groups}, input.options());
    auto inv_std = torch::empty({batch_size, num_groups}, input.options());

    dim3 block(256);
    dim3 grid(num_groups, batch_size);
    size_t shared_mem_size = 2 * block.x * sizeof(double);

    compute_mean_var_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        mean.data_ptr<float>(),
        inv_std.data_ptr<float>(),
        batch_size,
        num_groups,
        channels_per_group,
        height,
        width,
        eps
    );

    return {mean, inv_std};
}
"""

fused_lib = load_inline(
    name="fused_lib",
    cpp_sources="""
    #include <torch/extension.h>
    #include <vector>
    std::vector<torch::Tensor> compute_mean_var(torch::Tensor input, int num_groups, float eps);
    torch::Tensor fused_op(torch::Tensor input, torch::Tensor mean_tensor, torch::Tensor inv_std_tensor, torch::Tensor weight, torch::Tensor bias, torch::Tensor extra_scale, int output_h, int output_w, int pool_k, float clamp_min, float clamp_max);
    """,
    cuda_sources=hip_source,
    functions=["compute_mean_var", "fused_op"],
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
        
        input_h, input_w = x.size(2), x.size(3)
        output_h = (input_h - self.maxpool_kernel_size) // self.maxpool_kernel_size + 1
        output_w = (input_w - self.maxpool_kernel_size) // self.maxpool_kernel_size + 1

        mean, inv_std = fused_lib.compute_mean_var(x, self.num_groups, self.group_norm.eps)
        
        x = fused_lib.fused_op(
            x, mean, inv_std, 
            self.group_norm.weight, self.group_norm.bias, self.scale.view(-1),
            output_h, output_w, self.maxpool_kernel_size,
            self.clamp_min, self.clamp_max
        )
        
        return x

