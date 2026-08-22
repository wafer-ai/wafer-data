import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused HIP/ROCm kernel:
# Replace: tanh -> mul(scaling) -> add(bias) -> maxpool2d(k=4,s=4)
# with: maxpool2d on pre-activation conv output, then tanh+scale+bias.
# This is mathematically exact for scale>=0 (tanh is monotonic), and avoids
# materializing the large post-activation tensor.
os.environ.setdefault("CXX", "hipcc")

_cuda_src = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

__global__ void fused_pool_tanh_scale_bias_4x4_s4(
    const float* __restrict__ in,
    const float* __restrict__ bias_c,
    float* __restrict__ out,
    int N, int C, int H, int W,
    int Hout, int Wout,
    float scale)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = N * C * Hout * Wout;
    if (idx >= total) return;

    int ow = idx % Wout;
    int t = idx / Wout;
    int oh = t % Hout;
    t /= Hout;
    int c = t % C;
    int n = t / C;

    int ih0 = oh * 4;
    int iw0 = ow * 4;

    const float* p0 = in + ((n * C + c) * H + ih0) * W + iw0;

    // If scale < 0, monotonicity flips and we need min instead of max.
    float acc;
    if (scale >= 0.0f) {
        acc = -INFINITY;
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            const float* pr = p0 + r * W;
            float v0 = pr[0];
            float v1 = pr[1];
            float v2 = pr[2];
            float v3 = pr[3];
            acc = fmaxf(acc, v0);
            acc = fmaxf(acc, v1);
            acc = fmaxf(acc, v2);
            acc = fmaxf(acc, v3);
        }
    } else {
        acc = INFINITY;
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            const float* pr = p0 + r * W;
            float v0 = pr[0];
            float v1 = pr[1];
            float v2 = pr[2];
            float v3 = pr[3];
            acc = fminf(acc, v0);
            acc = fminf(acc, v1);
            acc = fminf(acc, v2);
            acc = fminf(acc, v3);
        }
    }

    float y = tanhf(acc) * scale + bias_c[c];
    out[idx] = y;
}

torch::Tensor fused_tanh_scale_bias_maxpool(torch::Tensor x, torch::Tensor bias, double scaling_factor) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA/HIP tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA/HIP tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be NCHW");

    x = x.contiguous();
    bias = bias.contiguous();

    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);

    TORCH_CHECK(bias.numel() == C, "bias must have numel == C");

    // Specialize benchmark config: k=4, stride=4
    const int k = 4;
    const int s = 4;

    TORCH_CHECK(H >= k && W >= k, "H/W too small for 4x4 pool");

    const int64_t Hout = (H - k) / s + 1;
    const int64_t Wout = (W - k) / s + 1;

    auto out = torch::empty({N, C, Hout, Wout}, x.options());

    int total = (int)(N * C * Hout * Wout);
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    float scale = (float)scaling_factor;

    auto stream = at::cuda::getDefaultCUDAStream();
    hipStream_t hip_stream = (hipStream_t)stream.stream();

    fused_pool_tanh_scale_bias_4x4_s4<<<blocks, threads, 0, hip_stream>>>(
        (const float*)x.data_ptr<float>(),
        (const float*)bias.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        (int)N, (int)C, (int)H, (int)W,
        (int)Hout, (int)Wout,
        scale);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tanh_scale_bias_maxpool", &fused_tanh_scale_bias_maxpool,
          "fused_tanh_scale_bias_maxpool (HIP)");
}
'''

_fused_ext = load_inline(
    name="fused_tanh_scale_bias_maxpool_ext",
    cpp_sources="",
    cuda_sources=_cuda_src,
    functions=None,
    with_cuda=True,
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """Optimized Model: conv2d + fused(pool + tanh + scale + bias)."""

    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = float(scaling_factor)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = self.conv(x)
        if self.pool_kernel_size != 4:
            # Safety fallback for non-benchmark configs
            x = torch.tanh(x)
            x = x * self.scaling_factor
            x = x + self.bias
            x = torch.nn.functional.max_pool2d(x, self.pool_kernel_size)
            return x
        return _fused_ext.fused_tanh_scale_bias_maxpool(x, self.bias, self.scaling_factor)


# Same generators as reference
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
