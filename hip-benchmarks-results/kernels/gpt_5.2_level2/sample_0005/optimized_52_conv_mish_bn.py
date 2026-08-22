import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Compile HIP extension (FP32)
os.environ.setdefault("CXX", "hipcc")

hip_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

namespace {

__device__ __forceinline__ float softplus_stable(float x) {
    // Stable softplus for FP32
    // softplus(x) = log(1 + exp(x))
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__device__ __forceinline__ float mish(float x) {
    float sp = softplus_stable(x);
    return x * tanhf(sp);
}

__global__ void mish_fwd_f4(const float* __restrict__ x, float* __restrict__ y, int64_t n4) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n4) return;
    const float4* x4p = reinterpret_cast<const float4*>(x);
    float4 v = x4p[tid];
    v.x = mish(v.x);
    v.y = mish(v.y);
    v.z = mish(v.z);
    v.w = mish(v.w);
    reinterpret_cast<float4*>(y)[tid] = v;
}

__global__ void mish_fwd_scalar(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    y[tid] = mish(x[tid]);
}

__global__ void mish_bn_inference_f4(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float* __restrict__ y,
    int C,
    int HW,
    float eps,
    int64_t n4)
{
    int64_t tid4 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid4 >= n4) return;

    int64_t base = tid4 * 4; // element index
    int c = (int)((base / HW) % C);

    float m = mean[c];
    float invstd = rsqrtf(var[c] + eps);
    float ww = w ? w[c] : 1.0f;
    float bb = b ? b[c] : 0.0f;

    float4 v = reinterpret_cast<const float4*>(x)[tid4];
    v.x = (mish(v.x) - m) * invstd * ww + bb;
    v.y = (mish(v.y) - m) * invstd * ww + bb;
    v.z = (mish(v.z) - m) * invstd * ww + bb;
    v.w = (mish(v.w) - m) * invstd * ww + bb;

    reinterpret_cast<float4*>(y)[tid4] = v;
}

__global__ void mish_bn_inference_scalar(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float* __restrict__ y,
    int C,
    int HW,
    float eps,
    int64_t n)
{
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int c = (int)((tid / HW) % C);
    float m = mean[c];
    float invstd = rsqrtf(var[c] + eps);
    float ww = w ? w[c] : 1.0f;
    float bb = b ? b[c] : 0.0f;

    float v = mish(x[tid]);
    y[tid] = (v - m) * invstd * ww + bb;
}

inline int64_t div_up_i64(int64_t a, int64_t b) { return (a + b - 1) / b; }

} // namespace

// Public API

torch::Tensor mish_hip(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "mish_hip: x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "mish_hip: x must be float32");
    auto y = torch::empty_like(x);

    auto n = x.numel();
    if (n == 0) return y;

    const int threads = 256;
    auto stream = at::cuda::getDefaultCUDAStream();

    if (x.is_contiguous() && (n % 4 == 0)) {
        int64_t n4 = n / 4;
        dim3 blocks((unsigned)div_up_i64(n4, threads));
        hipLaunchKernelGGL(mish_fwd_f4, blocks, dim3(threads), 0, stream,
                           (const float*)x.data_ptr<float>(), (float*)y.data_ptr<float>(), n4);
    } else {
        dim3 blocks((unsigned)div_up_i64(n, threads));
        hipLaunchKernelGGL(mish_fwd_scalar, blocks, dim3(threads), 0, stream,
                           (const float*)x.data_ptr<float>(), (float*)y.data_ptr<float>(), n);
    }

    return y;
}

torch::Tensor mish_bn_inference_hip(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    torch::Tensor mean,
    torch::Tensor var,
    double eps)
{
    TORCH_CHECK(x.is_cuda(), "mish_bn_inference_hip: x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "mish_bn_inference_hip: x must be float32");
    TORCH_CHECK(x.is_contiguous(), "mish_bn_inference_hip: x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "mish_bn_inference_hip: x must be NCHW");

    TORCH_CHECK(mean.is_cuda() && var.is_cuda(), "mish_bn_inference_hip: mean/var must be CUDA");
    TORCH_CHECK(mean.dtype() == torch::kFloat32 && var.dtype() == torch::kFloat32, "mish_bn_inference_hip: mean/var must be float32");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    int64_t HW = H * W;

    TORCH_CHECK(mean.numel() == C && var.numel() == C, "mish_bn_inference_hip: mean/var numel must equal C");
    TORCH_CHECK(w.numel() == C && b.numel() == C, "mish_bn_inference_hip: w/b numel must equal C");

    auto y = torch::empty_like(x);
    int64_t n = N * C * HW;
    if (n == 0) return y;

    const int threads = 256;
    auto stream = at::cuda::getDefaultCUDAStream();

    if ((n % 4 == 0) && (HW % 4 == 0)) {
        int64_t n4 = n / 4;
        dim3 blocks((unsigned)div_up_i64(n4, threads));
        hipLaunchKernelGGL(mish_bn_inference_f4, blocks, dim3(threads), 0, stream,
            (const float*)x.data_ptr<float>(),
            (const float*)w.data_ptr<float>(),
            (const float*)b.data_ptr<float>(),
            (const float*)mean.data_ptr<float>(),
            (const float*)var.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            (int)C,
            (int)HW,
            (float)eps,
            n4);
    } else {
        dim3 blocks((unsigned)div_up_i64(n, threads));
        hipLaunchKernelGGL(mish_bn_inference_scalar, blocks, dim3(threads), 0, stream,
            (const float*)x.data_ptr<float>(),
            (const float*)w.data_ptr<float>(),
            (const float*)b.data_ptr<float>(),
            (const float*)mean.data_ptr<float>(),
            (const float*)var.data_ptr<float>(),
            (float*)y.data_ptr<float>(),
            (int)C,
            (int)HW,
            (float)eps,
            n);
    }

    return y;
}

"""

_ext = load_inline(
    name="kb52_mish_bn_rocm",
    cpp_sources=hip_src,
    functions=["mish_hip", "mish_bn_inference_hip"],
    with_cuda=True,
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """Optimized: fuse Mish (x*tanh(softplus(x))) and BatchNorm in eval/inference.

    Training path falls back to PyTorch BN to preserve semantics.
    """

    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self._ext = _ext

    def forward(self, x):
        x = self.conv(x)
        # If BN is in eval mode, use fused Mish+BN inference kernel.
        if not self.bn.training:
            return self._ext.mish_bn_inference_hip(
                x,
                self.bn.weight,
                self.bn.bias,
                self.bn.running_mean,
                self.bn.running_var,
                float(self.bn.eps),
            )
        # Training: keep BN semantics; still speed up Mish by fusing softplus+tanh+mul.
        x = self._ext.mish_hip(x)
        x = self.bn(x)
        return x


# Keep I/O helpers identical to reference
batch_size = 64
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
