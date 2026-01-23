import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Ensure we build with hipcc on ROCm
os.environ.setdefault("CXX", "hipcc")

# Fused: (x - sub1) -> tanh -> (.. - sub2) -> avgpool2d(k, stride=k)
# Input: NCHW float32 contiguous
# Output: NCHW float32 contiguous with pooled spatial dims

_src = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <vector>

__device__ __forceinline__ float act_fused(float x, float sub1, float sub2) {
    // tanh(x - sub1) - sub2
    return tanhf(x - sub1) - sub2;
}

__global__ void fused_postpool_k2_kernel(
    const float* __restrict__ inp,
    float* __restrict__ out,
    int N, int C, int H, int W,
    float sub1, float sub2
) {
    // AvgPool2d(k=2, stride=2)
    const int outH = (H - 2) / 2 + 1;
    const int outW = (W - 2) / 2 + 1;
    const int64_t total = (int64_t)N * C * outH * outW;

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t linear = tid; linear < total; linear += stride) {
        int64_t t = linear;
        int ow = (int)(t % outW); t /= outW;
        int oh = (int)(t % outH); t /= outH;
        int c  = (int)(t % C);    t /= C;
        int n  = (int)t;

        int ih = oh * 2;
        int iw = ow * 2;

        int64_t base = (((int64_t)n * C + c) * H + ih) * W + iw;

        // Two float2 loads (2x2 window)
        const float2 r0 = *reinterpret_cast<const float2*>(inp + base);
        const float2 r1 = *reinterpret_cast<const float2*>(inp + base + W);

        float sum = 0.0f;
        sum += act_fused(r0.x, sub1, sub2);
        sum += act_fused(r0.y, sub1, sub2);
        sum += act_fused(r1.x, sub1, sub2);
        sum += act_fused(r1.y, sub1, sub2);

        out[linear] = sum * 0.25f;
    }
}

__global__ void fused_postpool_generic_kernel(
    const float* __restrict__ inp,
    float* __restrict__ out,
    int N, int C, int H, int W,
    int k,
    float sub1, float sub2
) {
    const int outH = (H - k) / k + 1;
    const int outW = (W - k) / k + 1;
    const int64_t total = (int64_t)N * C * outH * outW;

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    const float inv = 1.0f / (float)(k * k);

    for (int64_t linear = tid; linear < total; linear += stride) {
        int64_t t = linear;
        int ow = (int)(t % outW); t /= outW;
        int oh = (int)(t % outH); t /= outH;
        int c  = (int)(t % C);    t /= C;
        int n  = (int)t;

        int ih0 = oh * k;
        int iw0 = ow * k;
        int64_t base0 = (((int64_t)n * C + c) * H + ih0) * W + iw0;

        float sum = 0.0f;
        for (int dh = 0; dh < k; ++dh) {
            int64_t row = base0 + (int64_t)dh * W;
            for (int dw = 0; dw < k; ++dw) {
                sum += act_fused(inp[row + dw], sub1, sub2);
            }
        }
        out[linear] = sum * inv;
    }
}

torch::Tensor fused_postpool_hip(torch::Tensor inp, double sub1, double sub2, int64_t k) {
    TORCH_CHECK(inp.is_cuda(), "inp must be a CUDA/HIP tensor");
    TORCH_CHECK(inp.scalar_type() == at::kFloat, "inp must be float32");
    TORCH_CHECK(inp.dim() == 4, "inp must be NCHW");

    auto x = inp.contiguous();
    const int N = (int)x.size(0);
    const int C = (int)x.size(1);
    const int H = (int)x.size(2);
    const int W = (int)x.size(3);
    const int kk = (int)k;
    TORCH_CHECK(kk >= 1, "k must be >= 1");

    const int outH = (H - kk) / kk + 1;
    const int outW = (W - kk) / kk + 1;
    TORCH_CHECK(outH > 0 && outW > 0, "invalid pooling output size");

    auto out = torch::empty({N, C, outH, outW}, x.options());

    const int threads = 256;
    const int64_t total = (int64_t)N * C * outH * outW;
    int blocks = (int)((total + threads - 1) / threads);
    // Cap blocks to avoid absurd launch sizes; grid-stride loop handles the rest
    if (blocks > 131072) blocks = 131072;

    const float fsub1 = (float)sub1;
    const float fsub2 = (float)sub2;

    if (kk == 2) {
        hipLaunchKernelGGL(fused_postpool_k2_kernel, dim3(blocks), dim3(threads), 0, 0,
                           x.data_ptr<float>(), out.data_ptr<float>(), N, C, H, W, fsub1, fsub2);
    } else {
        hipLaunchKernelGGL(fused_postpool_generic_kernel, dim3(blocks), dim3(threads), 0, 0,
                           x.data_ptr<float>(), out.data_ptr<float>(), N, C, H, W, kk, fsub1, fsub2);
    }
    return out;
}
"""

_fused = load_inline(
    name="fused_postpool_46",
    cpp_sources=_src,
    functions=["fused_postpool_hip"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "-ffast-math"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """Optimized Model with fused (sub -> tanh -> sub -> avgpool) HIP kernel."""

    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = float(subtract1_value)
        self.subtract2_value = float(subtract2_value)
        self.kernel_size_pool = int(kernel_size_pool)

    def forward(self, x):
        x = self.conv(x)
        # fused: (x - sub1) -> tanh -> (.. - sub2) -> avgpool2d(k)
        return _fused.fused_postpool_hip(x, self.subtract1_value, self.subtract2_value, self.kernel_size_pool)


# Keep the same input generators for KernelBench
batch_size = 128
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
subtract1_value = 0.5
subtract2_value = 0.2
kernel_size_pool = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool]
