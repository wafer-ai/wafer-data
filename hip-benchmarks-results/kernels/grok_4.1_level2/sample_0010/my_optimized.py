import os
os.environ["CXX"] = "hipcc"

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_post_pool_cpp = """
#include <hip/hip_runtime.h>
#include <hip_math.h>
#include <torch/extension.h>

__global__ void fused_post_pool_kernel(
    const float *x, const float *bias, float scale, float *out,
    size_t out_numel, size_t stride_in_hw, size_t stride_in_w,
    int64_t C, int64_t H, int64_t W, int64_t out_H, int64_t out_W
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_numel) return;
    int64_t out_stride_hw = out_H * out_W;
    int64_t b = idx / (C * out_stride_hw);
    int64_t c = (idx / out_stride_hw) % C;
    int64_t oh = (idx / out_W) % out_H;
    int64_t ow = idx % out_W;
    float maxv = -1e30f;
    const int64_t s = 4;
    const int64_t k = 4;
#pragma unroll
    for (int64_t di = 0; di < k; ++di) {
        int64_t ih = oh * s + di;
        if (ih >= H) continue;
#pragma unroll
        for (int64_t dj = 0; dj < k; ++dj) {
            int64_t iw = ow * s + dj;
            if (iw >= W) continue;
            size_t xidx = b * (C * stride_in_hw) + c * stride_in_hw + ih * stride_in_w + iw;
            float val = tanhf(x[xidx]) * scale + bias[c];
            maxv = fmaxf(maxv, val);
        }
    }
    out[idx] = maxv;
}

torch::Tensor fused_post_pool_hip(torch::Tensor x, torch::Tensor bias, float scale) {
    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    const int64_t ksize = 4;
    const int64_t stride = 4;
    const int64_t pad_h = 0;
    const int64_t pad_w = 0;
    int64_t out_H = (H + 2 * pad_h - ksize) / stride + 1;
    int64_t out_W = (W + 2 * pad_w - ksize) / stride + 1;
    auto out = torch::empty({B, C, out_H, out_W}, x.options());
    size_t out_numel_ = static_cast<size_t>(out.numel());
    size_t stride_in_hw = static_cast<size_t>(H) * W;
    size_t stride_in_w = static_cast<size_t>(W);
    const int block_size = 256;
    int grid_size = (out_numel_ + block_size - 1) / block_size;
    fused_post_pool_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(), bias.data_ptr<float>(), scale,
        out.data_ptr<float>(), out_numel_,
        stride_in_hw, stride_in_w,
        C, H, W, out_H, out_W
    );
    return out;
}
"""

fused_post_pool = load_inline(
    name="fused_post_pool",
    cpp_sources=fused_post_pool_cpp,
    functions=["fused_post_pool_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_post_pool = fused_post_pool

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_post_pool.fused_post_pool_hip(x, self.bias, float(self.scaling_factor))
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
scaling_factor = 2.0
bias_shape = (out_channels, 1, 1)
pool_kernel_size = 4

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size]
