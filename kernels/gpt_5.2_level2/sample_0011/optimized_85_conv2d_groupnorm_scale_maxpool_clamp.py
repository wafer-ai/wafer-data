import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Compile HIP extension
os.environ.setdefault("CXX", "hipcc")

hip_src = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA/HIP tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")

__inline__ __device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset);
    }
    return v;
}

// One block computes stats for one (n, g)
__global__ void groupnorm_stats_kernel(
    const float* __restrict__ x,
    float* __restrict__ mean,
    float* __restrict__ invstd,
    int N, int C, int H, int W, int G,
    float eps)
{
    int ng = (int)blockIdx.x;
    int n = ng / G;
    int g = ng - n * G;
    int Cg = C / G;
    int HW = H * W;
    int64_t group_base = ((int64_t)n * C + (int64_t)g * Cg) * (int64_t)HW;

    float sum = 0.0f;
    float sumsq = 0.0f;

    int elements = Cg * HW;
    for (int i = (int)threadIdx.x; i < elements; i += (int)blockDim.x) {
        int c_in_g = i / HW;
        int hw = i - c_in_g * HW;
        float v = x[group_base + (int64_t)c_in_g * HW + hw];
        sum += v;
        sumsq += v * v;
    }

    // Block reduction (shared memory)
    __shared__ float sh_sum[256];
    __shared__ float sh_sumsq[256];

    int tid = (int)threadIdx.x;
    sh_sum[tid] = sum;
    sh_sumsq[tid] = sumsq;
    __syncthreads();

    for (int stride = ((int)blockDim.x) >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_sum[tid] += sh_sum[tid + stride];
            sh_sumsq[tid] += sh_sumsq[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float m = sh_sum[0] / (float)elements;
        float v = sh_sumsq[0] / (float)elements - m * m;
        v = v > 0.0f ? v : 0.0f;
        mean[ng] = m;
        invstd[ng] = rsqrtf(v + eps);
    }
}

__device__ __forceinline__ float clampf(float x, float lo, float hi) {
    x = x < lo ? lo : x;
    x = x > hi ? hi : x;
    return x;
}

// Fuses: groupnorm (apply) + scale + maxpool(k=stride=4) + clamp
__global__ void fused_gn_scale_pool_clamp_k4_kernel(
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ invstd,
    const float* __restrict__ gn_weight,
    const float* __restrict__ gn_bias,
    const float* __restrict__ scale,
    float* __restrict__ out,
    int N, int C, int H, int W, int G,
    int Hp, int Wp,
    float clamp_min, float clamp_max)
{
    int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t total = (int64_t)N * C * Hp * Wp;
    if (idx >= total) return;

    int pw = (int)(idx % Wp);
    int64_t t = idx / Wp;
    int ph = (int)(t % Hp);
    t /= Hp;
    int c = (int)(t % C);
    int n = (int)(t / C);

    int Cg = C / G;
    int g = c / Cg;

    float m = mean[n * G + g];
    float inv = invstd[n * G + g];

    float sc = scale[c];
    float w = gn_weight[c];
    float b0 = gn_bias[c];

    // y = ((x - m) * inv * w + b0) * sc = x*a + b
    float a = inv * w * sc;
    float b = b0 * sc - m * a;

    int h0 = ph * 4;
    int w0 = pw * 4;

    const float* base = x + (((int64_t)n * C + c) * H + h0) * (int64_t)W + w0;

    float vmax = -INFINITY;

    // 4 rows, each row load float4 (w0 is multiple of 4)
    #pragma unroll
    for (int kh = 0; kh < 4; ++kh) {
        const float4 v4 = *reinterpret_cast<const float4*>(base + (int64_t)kh * W);
        float y0 = fmaf(v4.x, a, b);
        float y1 = fmaf(v4.y, a, b);
        float y2 = fmaf(v4.z, a, b);
        float y3 = fmaf(v4.w, a, b);
        vmax = fmaxf(vmax, y0);
        vmax = fmaxf(vmax, y1);
        vmax = fmaxf(vmax, y2);
        vmax = fmaxf(vmax, y3);
    }

    vmax = clampf(vmax, clamp_min, clamp_max);
    out[idx] = vmax;
}

torch::Tensor fused_gn_scale_maxpool_clamp_hip(
    torch::Tensor x,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    torch::Tensor scale,
    int64_t num_groups,
    double eps,
    double clamp_min,
    double clamp_max)
{
    CHECK_CUDA(x);
    CHECK_CUDA(gn_weight);
    CHECK_CUDA(gn_bias);
    CHECK_CUDA(scale);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(gn_weight);
    CHECK_CONTIGUOUS(gn_bias);
    CHECK_CONTIGUOUS(scale);
    CHECK_FLOAT(x);
    CHECK_FLOAT(gn_weight);
    CHECK_FLOAT(gn_bias);
    CHECK_FLOAT(scale);

    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int G = (int)num_groups;
    TORCH_CHECK(C % G == 0, "C must be divisible by num_groups");

    // Fixed maxpool kernel=4, stride=4 (matches benchmark)
    const int pool_k = 4;
    int Hp = (H - pool_k) / pool_k + 1;
    int Wp = (W - pool_k) / pool_k + 1;

    auto opts = x.options();
    auto mean = torch::empty({N * G}, opts);
    auto invstd = torch::empty({N * G}, opts);
    auto out = torch::empty({N, C, Hp, Wp}, opts);

    hipStream_t stream = (hipStream_t)at::cuda::getDefaultCUDAStream();

    const int stats_threads = 256;
    dim3 stats_grid(N * G);
    dim3 stats_block(stats_threads);

    hipLaunchKernelGGL(
        groupnorm_stats_kernel,
        stats_grid,
        stats_block,
        0,
        stream,
        (const float*)x.data_ptr<float>(),
        (float*)mean.data_ptr<float>(),
        (float*)invstd.data_ptr<float>(),
        N, C, H, W, G,
        (float)eps);

    int64_t total = (int64_t)N * C * Hp * Wp;
    const int threads = 256;
    dim3 grid((unsigned int)((total + threads - 1) / threads));
    dim3 block(threads);

    hipLaunchKernelGGL(
        fused_gn_scale_pool_clamp_k4_kernel,
        grid,
        block,
        0,
        stream,
        (const float*)x.data_ptr<float>(),
        (const float*)mean.data_ptr<float>(),
        (const float*)invstd.data_ptr<float>(),
        (const float*)gn_weight.data_ptr<float>(),
        (const float*)gn_bias.data_ptr<float>(),
        (const float*)scale.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        N, C, H, W, G,
        Hp, Wp,
        (float)clamp_min, (float)clamp_max);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_gn_scale_maxpool_clamp_hip", &fused_gn_scale_maxpool_clamp_hip,
          "Fused GroupNorm+Scale+MaxPool(k=4)+Clamp (HIP)");
}
"""

# Use a stable name to allow caching across runs
_ext = load_inline(
    name="fused_gn_scale_pool_clamp_ext",
    cpp_sources=hip_src,
    functions=None,
    extra_cflags=["-O3"],
    with_cuda=False,
    verbose=False,
)


class ModelNew(nn.Module):
    """Optimized model: keep Conv2d (MIOpen), fuse GroupNorm+Scale+MaxPool+Clamp into custom HIP kernels."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_groups,
        scale_shape,
        maxpool_kernel_size,
        clamp_min,
        clamp_max,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        # kept for API parity; we fuse maxpool
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

    def forward(self, x):
        x = self.conv(x)
        # Ensure contiguous for the custom kernel
        if not x.is_contiguous():
            x = x.contiguous()
        w = self.group_norm.weight
        b = self.group_norm.bias
        if not w.is_contiguous():
            w = w.contiguous()
        if not b.is_contiguous():
            b = b.contiguous()
        sc = self.scale
        if not sc.is_contiguous():
            sc = sc.contiguous()
        # This benchmark uses maxpool k=4; enforce to avoid silent mismatch
        k = self.maxpool.kernel_size
        if isinstance(k, tuple):
            k = k[0]
        if k != 4:
            # Fallback to reference if someone changes params
            x = self.group_norm(x)
            x = x * self.scale
            x = self.maxpool(x)
            return torch.clamp(x, self.clamp_min, self.clamp_max)
        return _ext.fused_gn_scale_maxpool_clamp_hip(
            x, w, b, sc.view(-1).contiguous(), self.group_norm.num_groups, self.group_norm.eps, self.clamp_min, self.clamp_max
        )


# Reference input generators
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
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        num_groups,
        scale_shape,
        maxpool_kernel_size,
        clamp_min,
        clamp_max,
    ]
