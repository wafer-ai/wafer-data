import os
os.environ["CXX"] = "hipcc"
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_source = """
#include <hip/hip_runtime.h>
#include <cmath>

__global__ void compute_group_stats_kernel(const float* input, float* means, float* vars, int N, int C, int H, int W, int num_groups, int vol, float eps) {
    int bin = blockIdx.x;
    if (bin >= N * num_groups) return;
    int n = bin / num_groups;
    int g = bin % num_groups;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    for (int pos = tid; pos < vol; pos += stride) {
        int c_local = pos / (H * W);
        int hw = pos % (H * W);
        int h = hw / W;
        int w = hw % W;
        int c = g * (C / num_groups) + c_local;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = input[idx];
        local_sum += val;
        local_sumsq += val * val;
    }
    __shared__ float sh_sum[1024];
    __shared__ float sh_sumsq[1024];
    sh_sum[tid] = local_sum;
    sh_sumsq[tid] = local_sumsq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh_sum[tid] += sh_sum[tid + s];
            sh_sumsq[tid] += sh_sumsq[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        float sumv = sh_sum[0];
        float sumsqv = sh_sumsq[0];
        float mean = sumv / static_cast<float>(vol);
        float var = (sumsqv / static_cast<float>(vol)) - mean * mean + eps;
        means[bin] = mean;
        vars[bin] = 1.0f / sqrtf(var);
    }
}

__global__ void apply_norm_scale_kernel(const float* input, const float* gn_weight, const float* gn_bias, const float* scale,
    const float* means, const float* vars, float* output,
    int N, int C, int H, int W, int num_groups) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H * W) return;
    int n = idx / (C * H * W);
    int rest = idx % (C * H * W);
    int c = rest / (H * W);
    int g = c / (C / num_groups);
    float mean = means[n * num_groups + g];
    float rstd = vars[n * num_groups + g];
    float val = input[idx];
    float normed = rstd * (val - mean) * gn_weight[c] + gn_bias[c];
    normed *= scale[c];
    output[idx] = normed;
}

__global__ void maxpool_clamp_kernel(const float* input, float* output, float clamp_min, float clamp_max,
    int N, int C, int Hi, int Wi, int Ho, int Wo, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * Ho * Wo;
    if (idx >= total) return;
    int n = idx / (C * Ho * Wo);
    int rest = idx % (C * Ho * Wo);
    int c = rest / (Ho * Wo);
    int rest2 = rest % (Ho * Wo);
    int ho = rest2 / Wo;
    int wo = rest2 % Wo;
    float mval = -1e30f;
    for (int dh = 0; dh < K; ++dh) {
        int hi = ho * K + dh;
        if (hi >= Hi) continue;
        for (int dw = 0; dw < K; ++dw) {
            int wi = wo * K + dw;
            if (wi >= Wi) continue;
            int in_idx = ((n * C + c) * Hi + hi) * Wi + wi;
            mval = fmaxf(mval, input[in_idx]);
        }
    }
    float val = fmaxf(clamp_min, fminf(clamp_max, mval));
    int out_idx = ((n * C + c) * Ho + ho) * Wo + wo;
    output[out_idx] = val;
}

torch::Tensor fused_norm_scale_hip(torch::Tensor input, torch::Tensor gn_weight, torch::Tensor gn_bias, torch::Tensor scale, int64_t num_groups_, float eps) {
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int num_groups = static_cast<int>(num_groups_);
    int Cg = static_cast<int>(C) / num_groups;
    int spatial = static_cast<int>(H) * static_cast<int>(W);
    int vol = Cg * spatial;
    auto options = input.options();
    int64_t num_bins = N * num_groups_;
    torch::Tensor means = torch::empty({num_bins}, options);
    torch::Tensor vars = torch::empty({num_bins}, options);
    const int bs = 1024;
    int grid_stats = static_cast<int>(num_bins);
    compute_group_stats_kernel<<<grid_stats, bs>>>(input.data_ptr<float>(), means.data_ptr<float>(), vars.data_ptr<float>(),
        static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W), num_groups, vol, eps);
    (void)hipDeviceSynchronize();
    torch::Tensor output = torch::empty_like(input);
    int64_t total = N * C * H * W;
    int grid_apply = static_cast<int>((total + bs - 1) / bs);
    apply_norm_scale_kernel<<<grid_apply, bs>>>(input.data_ptr<float>(), gn_weight.data_ptr<float>(), gn_bias.data_ptr<float>(), scale.data_ptr<float>(),
        means.data_ptr<float>(), vars.data_ptr<float>(), output.data_ptr<float>(),
        static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W), num_groups);
    (void)hipDeviceSynchronize();
    return output;
}

torch::Tensor maxpool_clamp_hip(torch::Tensor input, float clamp_min, float clamp_max, int64_t kernel_size_) {
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t Hi = input.size(2);
    int64_t Wi = input.size(3);
    int K = static_cast<int>(kernel_size_);
    int Ho = static_cast<int>((Hi - K) / K + 1);
    int Wo = static_cast<int>((Wi - K) / K + 1);
    torch::Tensor output = torch::empty({N, C, (int64_t)Ho, (int64_t)Wo}, input.options());
    const int bs = 1024;
    int64_t total_out = N * C * (int64_t)Ho * Wo;
    int grid_pool = static_cast<int>((total_out + bs - 1) / bs);
    maxpool_clamp_kernel<<<grid_pool, bs>>>(input.data_ptr<float>(), output.data_ptr<float>(), clamp_min, clamp_max,
        static_cast<int>(N), static_cast<int>(C), static_cast<int>(Hi), static_cast<int>(Wi), Ho, Wo, K);
    (void)hipDeviceSynchronize();
    return output;
}
"""

model_ops = load_inline(
    name="model_ops",
    cpp_sources=cpp_source,
    functions=["fused_norm_scale_hip", "maxpool_clamp_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.gn_weight = nn.Parameter(torch.ones(out_channels))
        self.gn_bias = nn.Parameter(torch.zeros(out_channels))
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.num_groups = num_groups
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.model_ops = model_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.model_ops.fused_norm_scale_hip(x, self.gn_weight, self.gn_bias, self.scale, self.num_groups, 1e-5)
        x = self.model_ops.maxpool_clamp_hip(x, self.clamp_min, self.clamp_max, self.maxpool_kernel_size)
        return x
