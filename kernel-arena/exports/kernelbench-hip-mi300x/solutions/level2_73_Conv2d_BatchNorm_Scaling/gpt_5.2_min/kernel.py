import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ.setdefault("CXX", "hipcc")

hip_src = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>

// Vectorized loop (unroll 4) over contiguous FP32 values.
// y = ((x - mean)/sqrt(var+eps) * weight + bias) * scale
__global__ void bn_scale_nchw_f32x4_kernel(
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int N, int C, int H, int W,
    float eps,
    float scale)
{
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = N * C * H * W;

    int i = tid * 4;
    int stride = blockDim.x * gridDim.x * 4;

    for (; i + 3 < total; i += stride) {
        #pragma unroll
        for (int lane = 0; lane < 4; lane++) {
            int idx = i + lane;

            int t = idx / W;
            t /= H;
            int c = t % C;

            float xm = x[idx];
            float inv_std = rsqrtf(var[c] + eps);
            float out = (xm - mean[c]) * inv_std;
            out = out * weight[c] + bias[c];
            y[idx] = out * scale;
        }
    }

    for (; i < total; i += stride) {
        int idx = i;
        int t = idx / W;
        t /= H;
        int c = t % C;

        float xm = x[idx];
        float inv_std = rsqrtf(var[c] + eps);
        float out = (xm - mean[c]) * inv_std;
        out = out * weight[c] + bias[c];
        y[idx] = out * scale;
    }
}

torch::Tensor bn_scale_eval_hip(torch::Tensor x,
                               torch::Tensor running_mean,
                               torch::Tensor running_var,
                               torch::Tensor weight,
                               torch::Tensor bias,
                               double eps,
                               double scale) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA/HIP tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto y = torch::empty_like(x);

    const int N = (int)x.size(0);
    const int C = (int)x.size(1);
    const int H = (int)x.size(2);
    const int W = (int)x.size(3);

    int total = N * C * H * W;

    const int threads = 256;
    int blocks = (total + (threads * 4) - 1) / (threads * 4);
    if (blocks > 4096) blocks = 4096;

    hipLaunchKernelGGL(bn_scale_nchw_f32x4_kernel,
                      dim3(blocks), dim3(threads), 0, 0,
                      (const float*)x.data_ptr<float>(),
                      (const float*)running_mean.data_ptr<float>(),
                      (const float*)running_var.data_ptr<float>(),
                      (const float*)weight.data_ptr<float>(),
                      (const float*)bias.data_ptr<float>(),
                      (float*)y.data_ptr<float>(),
                      N, C, H, W,
                      (float)eps,
                      (float)scale);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bn_scale_eval_hip", &bn_scale_eval_hip, "Fused BatchNorm(eval)+Scale (HIP)");
}
"""

bn_scale_ext = load_inline(
    name="bn_scale_ext_73",
    cpp_sources="",
    cuda_sources=hip_src,
    functions=None,
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """Optimized: keep Conv2d via MIOpen, fuse BatchNorm(eval)+scaling into one HIP kernel."""
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        x = self.conv(x)
        if not x.is_contiguous():
            x = x.contiguous()
        return bn_scale_ext.bn_scale_eval_hip(
            x,
            self.bn.running_mean.contiguous(),
            self.bn.running_var.contiguous(),
            self.bn.weight.contiguous(),
            self.bn.bias.contiguous(),
            float(self.bn.eps),
            float(self.scaling_factor),
        )


def get_inputs():
    batch_size = 128
    in_channels = 8
    height, width = 128, 128
    return [torch.rand(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)]


def get_init_inputs():
    in_channels = 8
    out_channels = 64
    kernel_size = 3
    scaling_factor = 2.0
    return [in_channels, out_channels, kernel_size, scaling_factor]
