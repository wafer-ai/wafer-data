import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Force HIP compilation on ROCm
os.environ.setdefault("CXX", "hipcc")

# Fused InstanceNorm2d (affine=False, track_running_stats=False) + divide-by-constant
# Assumes NCHW contiguous FP32.
instancenorm_div_cpp_source = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <c10/hip/HIPStream.h>

namespace {

__global__ void instancenorm_div_fwd_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int S,                 // H*W
    float eps,
    float inv_div          // 1.0f / divide_by
) {
    // One block per (N,C)
    const int group = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;

    // Shared reduction buffers
    __shared__ float sh_sum[256];
    __shared__ float sh_sumsq[256];

    const float* xg = x + ((int64_t)group) * (int64_t)S;
    float* yg = y + ((int64_t)group) * (int64_t)S;

    float sum = 0.0f;
    float sumsq = 0.0f;

    // Stride over spatial elements
    for (int i = tid; i < S; i += (int)blockDim.x) {
        float v = xg[i];
        sum += v;
        sumsq = fmaf(v, v, sumsq);
    }

    sh_sum[tid] = sum;
    sh_sumsq[tid] = sumsq;
    __syncthreads();

    // Reduce within block (blockDim must be 256)
    for (int offset = 128; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sh_sum[tid] += sh_sum[tid + offset];
            sh_sumsq[tid] += sh_sumsq[tid + offset];
        }
        __syncthreads();
    }

    float mean = sh_sum[0] / (float)S;
    float var = sh_sumsq[0] / (float)S - mean * mean;
    var = var > 0.0f ? var : 0.0f;
    float invstd = rsqrtf(var + eps);
    float scale = invstd * inv_div;

    // Normalize + divide
    for (int i = tid; i < S; i += (int)blockDim.x) {
        float v = xg[i];
        yg[i] = (v - mean) * scale;
    }
}

} // namespace

torch::Tensor instancenorm_div_hip(torch::Tensor x, double eps, double divide_by) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA/HIP tensor");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be NCHW 4D");

    auto x_contig = x.contiguous();
    auto y = torch::empty_like(x_contig);

    const int64_t N = x_contig.size(0);
    const int64_t C = x_contig.size(1);
    const int64_t H = x_contig.size(2);
    const int64_t W = x_contig.size(3);
    const int S = (int)(H * W);
    const int groups = (int)(N * C);

    constexpr int threads = 256;
    dim3 block(threads);
    dim3 grid(groups);

    hipStream_t stream = c10::hip::getCurrentHIPStream().stream();

    instancenorm_div_fwd_kernel<<<grid, block, 0, stream>>>(
        (const float*)x_contig.data_ptr<float>(),
        (float*)y.data_ptr<float>(),
        S,
        (float)eps,
        1.0f / (float)divide_by
    );

    return y;
}
"""

instancenorm_div_ext = load_inline(
    name="instancenorm_div_ext",
    cpp_sources=instancenorm_div_cpp_source,
    functions=["instancenorm_div_hip"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """Optimized: keep Conv2d, fuse InstanceNorm2d + divide into one HIP kernel."""

    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # Match InstanceNorm2d defaults
        self.eps = 1e-5
        self.divide_by = float(divide_by)
        self._fused = instancenorm_div_ext

    def forward(self, x):
        x = self.conv(x)
        # x is [N,C,H,W] FP32
        return self._fused.instancenorm_div_hip(x, self.eps, self.divide_by)
