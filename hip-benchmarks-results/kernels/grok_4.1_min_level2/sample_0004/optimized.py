import os
os.environ["CXX"] = "hipcc"

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

post_cpp_source = """
#include <hip/hip_runtime.h>
#include <cmath>

__global__ void fused_postpool_kernel(const float* in, float* out, float s1, float s2, size_t B, size_t C, size_t H, size_t W) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t out_H = H / 2;
    size_t out_W = W / 2;
    size_t out_numel = B * C * out_H * out_W;
    if (idx >= out_numel) return;
    size_t temp = idx;
    size_t b = temp / (C * out_H * out_W);
    temp %= (C * out_H * out_W);
    size_t c = temp / (out_H * out_W);
    temp %= (out_H * out_W);
    size_t oh = temp / out_W;
    size_t ow = temp % out_W;
    size_t ih = oh * 2;
    size_t iw = ow * 2;
    float sum = 0.0f;
    for (int di = 0; di < 2; ++di) {
        size_t ihh = ih + di;
        for (int dj = 0; dj < 2; ++dj) {
            size_t iww = iw + dj;
            size_t base_idx = ((b * C + c) * H + ihh) * W + iww;
            float val = tanhf(in[base_idx] - s1) - s2;
            sum += val;
        }
    }
    out[idx] = sum / 4.0f;
}

torch::Tensor fused_postpool_hip(torch::Tensor x, float s1, float s2) {
    auto sizes = x.sizes();
    size_t B = sizes[0];
    size_t C = sizes[1];
    size_t H = sizes[2];
    size_t W = sizes[3];
    size_t out_H = H / 2;
    size_t out_W = W / 2;
    auto out = torch::empty({int64_t(B), int64_t(C), int64_t(out_H), int64_t(out_W)}, x.options());
    size_t out_numel = B * C * out_H * out_W;
    const size_t block_size = 1024;
    size_t num_blocks = (out_numel + block_size - 1) / block_size;
    fused_postpool_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), s1, s2, B, C, H, W);
    return out;
}
"""

post_fused = load_inline(
    name="post_fused",
    cpp_sources=post_cpp_source,
    functions=["fused_postpool_hip"],
    verbose=True,
)

batch_size = 128
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
subtract1_value = 0.5
subtract2_value = 0.2
kernel_size_pool = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool]

class ModelNew(nn.Module):
    """
    Model that performs a convolution, subtraction, tanh activation, subtraction and average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.fused_post = post_fused

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_post.fused_postpool_hip(x, self.subtract1_value, self.subtract2_value)
        return x
