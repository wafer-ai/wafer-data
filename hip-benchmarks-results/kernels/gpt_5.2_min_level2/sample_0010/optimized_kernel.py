import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ.setdefault("CXX", "hipcc")

hip_src = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// Avoid relying on device tanhf libcalls (can be missing in some toolchains).
// Use an exp-based tanh approximation that is accurate enough for FP32.
__device__ __forceinline__ float fast_tanh(float x) {
    float ax = fabsf(x);
    // tanh(x) = sign(x) * (1 - e^{-2|x|}) / (1 + e^{-2|x|})
    float e = expf(-2.0f * ax);
    float t = (1.0f - e) / (1.0f + e);
    return copysignf(t, x);
}

template<int POOL_K>
__global__ void fused_tanh_scale_bias_maxpool_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int N, int C, int H, int W,
    float scale,
    int OH, int OW)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = N * C * OH * OW;
    if (idx >= total) return;

    int ow = idx % OW;
    int t = idx / OW;
    int oh = t % OH;
    t /= OH;
    int c = t % C;
    int n = t / C;

    int base_h = oh * POOL_K;
    int base_w = ow * POOL_K;

    const int HW = H * W;
    const float* x_nc = x + (n * C + c) * HW;

    float b = bias[c];
    float m = -INFINITY;

    #pragma unroll
    for (int ph = 0; ph < POOL_K; ++ph) {
        int ih = base_h + ph;
        const float* row = x_nc + ih * W + base_w;
        #pragma unroll
        for (int pw = 0; pw < POOL_K; ++pw) {
            float v = row[pw];
            v = fast_tanh(v);
            v = v * scale + b;
            m = v > m ? v : m;
        }
    }

    out[idx] = m;
}

torch::Tensor fused_tanh_scale_bias_maxpool_hip(torch::Tensor x, torch::Tensor bias, double scale, int64_t pool_k) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA/HIP tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA/HIP tensor");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, "x must be float32");
    TORCH_CHECK(bias.scalar_type() == at::ScalarType::Float, "bias must be float32");

    x = x.contiguous();
    bias = bias.contiguous().view({-1});

    const int N = (int)x.size(0);
    const int C = (int)x.size(1);
    const int H = (int)x.size(2);
    const int W = (int)x.size(3);

    const int k = (int)pool_k;
    TORCH_CHECK(k > 0, "pool_k must be > 0");
    TORCH_CHECK((int)bias.numel() == C, "bias must broadcast to channels");
    TORCH_CHECK(H >= k && W >= k, "pool kernel larger than input");

    const int OH = (H - k) / k + 1;
    const int OW = (W - k) / k + 1;

    auto out = torch::empty({N, C, OH, OW}, x.options());

    const int total = N * C * OH * OW;
    constexpr int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    hipStream_t stream = (hipStream_t)at::cuda::getDefaultCUDAStream();
    float fscale = (float)scale;

    if (k == 2) {
        hipLaunchKernelGGL((fused_tanh_scale_bias_maxpool_kernel<2>), dim3(blocks), dim3(threads), 0, stream,
            (const float*)x.data_ptr<float>(), (const float*)bias.data_ptr<float>(), (float*)out.data_ptr<float>(),
            N, C, H, W, fscale, OH, OW);
    } else if (k == 3) {
        hipLaunchKernelGGL((fused_tanh_scale_bias_maxpool_kernel<3>), dim3(blocks), dim3(threads), 0, stream,
            (const float*)x.data_ptr<float>(), (const float*)bias.data_ptr<float>(), (float*)out.data_ptr<float>(),
            N, C, H, W, fscale, OH, OW);
    } else if (k == 4) {
        hipLaunchKernelGGL((fused_tanh_scale_bias_maxpool_kernel<4>), dim3(blocks), dim3(threads), 0, stream,
            (const float*)x.data_ptr<float>(), (const float*)bias.data_ptr<float>(), (float*)out.data_ptr<float>(),
            N, C, H, W, fscale, OH, OW);
    } else if (k == 5) {
        hipLaunchKernelGGL((fused_tanh_scale_bias_maxpool_kernel<5>), dim3(blocks), dim3(threads), 0, stream,
            (const float*)x.data_ptr<float>(), (const float*)bias.data_ptr<float>(), (float*)out.data_ptr<float>(),
            N, C, H, W, fscale, OH, OW);
    } else {
        TORCH_CHECK(false, "Unsupported pool_k for fused kernel: ", k);
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tanh_scale_bias_maxpool_hip", &fused_tanh_scale_bias_maxpool_hip,
          "Fused tanh->scale->bias->maxpool (HIP)");
}
"""

fused_ext = load_inline(
    name="fused_tanh_scale_bias_maxpool_ext",
    cpp_sources=hip_src,
    functions=None,
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = float(scaling_factor)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.pool_kernel_size = int(pool_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        return fused_ext.fused_tanh_scale_bias_maxpool_hip(x, self.bias, self.scaling_factor, self.pool_kernel_size)


def get_inputs():
    return [torch.rand(128, 8, 256, 256, device="cuda", dtype=torch.float32)]


def get_init_inputs():
    in_channels = 8
    out_channels = 64
    kernel_size = 3
    scaling_factor = 2.0
    bias_shape = (out_channels, 1, 1)
    pool_kernel_size = 4
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size]
