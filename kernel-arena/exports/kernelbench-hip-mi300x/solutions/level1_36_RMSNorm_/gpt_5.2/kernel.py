import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Compile with hipcc for ROCm
os.environ.setdefault("CXX", "hipcc")

# Fused RMSNorm over dim=1 (feature/channel dim):
#   rms = sqrt(mean(x^2, dim=1, keepdim=True) + eps)
#   y = x / rms
# We fuse reduction + normalization into a single HIP kernel to avoid intermediates.

rmsnorm_cpp_source = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
  #include <ATen/hip/HIPContext.h>
#endif

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA/HIP tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")

__global__ void rmsnorm_fwd64_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int HW,
    int strideB,
    float eps)
{
    int p = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int b = (int)blockIdx.y;
    if (p >= HW) return;

    int base = b * strideB + p;
    float sumsq = 0.0f;

    #pragma unroll
    for (int f = 0; f < 64; ++f) {
        float v = x[base + f * HW];
        sumsq = fmaf(v, v, sumsq);
    }

    float mean = sumsq * (1.0f / 64.0f);
    float inv_rms = rsqrtf(mean + eps);

    #pragma unroll
    for (int f = 0; f < 64; ++f) {
        float v = x[base + f * HW];
        y[base + f * HW] = v * inv_rms;
    }
}

__global__ void rmsnorm_fwd_generic_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int F,
    int HW,
    int strideB,
    float invF,
    float eps)
{
    int p = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int b = (int)blockIdx.y;
    if (p >= HW) return;

    int base = b * strideB + p;
    float sumsq = 0.0f;

    for (int f = 0; f < F; ++f) {
        float v = x[base + f * HW];
        sumsq = fmaf(v, v, sumsq);
    }

    float mean = sumsq * invF;
    float inv_rms = rsqrtf(mean + eps);

    for (int f = 0; f < F; ++f) {
        float v = x[base + f * HW];
        y[base + f * HW] = v * inv_rms;
    }
}

torch::Tensor rmsnorm_hip(torch::Tensor x, double eps) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    CHECK_FLOAT(x);
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims (B, F, ...)");

    const int B = (int)x.size(0);
    const int F = (int)x.size(1);
    TORCH_CHECK(B > 0 && F > 0, "Invalid shapes");

    // Flatten remaining dims into HW.
    int64_t HW64 = x.numel() / ((int64_t)B * (int64_t)F);
    TORCH_CHECK(HW64 <= INT32_MAX, "HW too large");
    const int HW = (int)HW64;

    auto y = torch::empty_like(x);

    const int strideB = F * HW;
    const int threads = 256;
    dim3 block(threads);
    dim3 grid((HW + threads - 1) / threads, B, 1);

    hipStream_t stream = 0;
    #if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
      // Use PyTorch's current HIP stream
      stream = at::hip::getDefaultHIPStream();
    #endif

    if (F == 64) {
        hipLaunchKernelGGL(rmsnorm_fwd64_kernel, grid, block, 0, stream,
                           (const float*)x.data_ptr<float>(), (float*)y.data_ptr<float>(),
                           HW, strideB, (float)eps);
    } else {
        const float invF = 1.0f / (float)F;
        hipLaunchKernelGGL(rmsnorm_fwd_generic_kernel, grid, block, 0, stream,
                           (const float*)x.data_ptr<float>(), (float*)y.data_ptr<float>(),
                           F, HW, strideB, invF, (float)eps);
    }

    return y;
}
"""

rmsnorm_ext = load_inline(
    name="rmsnorm_hip_ext",
    cpp_sources=rmsnorm_cpp_source,
    functions=["rmsnorm_hip"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """Optimized RMSNorm using a fused HIP kernel (FP32)."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self._ext = rmsnorm_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        return self._ext.rmsnorm_hip(x, float(self.eps))


# KernelBench interface
batch_size = 112
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]


def get_init_inputs():
    return [features]
