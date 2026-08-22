import os
os.environ["CXX"] = "hipcc"
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

postpool_cpp = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>
#include <limits>

__global__ void post_pool_kernel(const float* inp, float scale, const float* bias, float* out, int B, int C, int H, int W, int Ph, int Pw) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * Ph * Pw;
    if (idx >= total) return;
    int temp = idx / (Ph * Pw);
    int b = temp / C;
    int c = temp % C;
    int phw = idx % (Ph * Pw);
    int ph = phw / Pw;
    int pw = phw % Pw;
    float maxv = -std::numeric_limits<float>::infinity();
    for (int doh = 0; doh < 4; ++doh) {
        int oh = ph * 4 + doh;
        if (oh >= H) continue;
        for (int dow = 0; dow < 4; ++dow) {
            int ow = pw * 4 + dow;
            if (ow >= W) continue;
            int fidx = ((b * C + c) * H + oh) * W + ow;
            float tempv = tanhf(inp[fidx]) * scale;
            if (tempv > maxv) maxv = tempv;
        }
    }
    int oidx = ((b * C + c) * Ph + ph) * Pw + pw;
    out[oidx] = maxv + bias[c];
}

torch::Tensor post_pool_hip(torch::Tensor inp, float scale, torch::Tensor bias) {
    auto desc = inp.sizes();
    int64_t B = desc[0];
    int64_t C = desc[1];
    int64_t H = desc[2];
    int64_t W = desc[3];
    int64_t Ph = (H - 4) / 4 + 1;
    int64_t Pw = (W - 4) / 4 + 1;
    torch::Tensor out = torch::zeros({B, C, Ph, Pw}, inp.options());
    int64_t total = B * C * Ph * Pw;
    const int bsize = 256;
    const int64_t nblocks = (total + bsize - 1) / bsize;
    post_pool_kernel<<<nblocks, bsize>>>(
        inp.data_ptr<float>(),
        scale,
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        (int)B, (int)C, (int)H, (int)W, (int)Ph, (int)Pw
    );
    return out;
}
"""

postpool_module = load_inline(
    name="postpool",
    cpp_sources=postpool_cpp,
    functions=["post_pool_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=True)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.postpool_module = postpool_module

    def forward(self, x):
        conv_out = self.conv(x)
        pooled = self.postpool_module.post_pool_hip(conv_out, float(self.scaling_factor), self.bias)
        return pooled

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
