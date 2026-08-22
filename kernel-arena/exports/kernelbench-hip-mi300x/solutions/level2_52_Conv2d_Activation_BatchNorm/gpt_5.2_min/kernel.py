import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Ensure ROCm uses hipcc
os.environ.setdefault("CXX", "hipcc")

hip_src = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

__device__ __forceinline__ float softplus_stable(float x) {
    // softplus(x)=log1p(exp(x)) with fp32 stability
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__global__ void fused_act_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int stride = (int)(blockDim.x * gridDim.x);
    for (int i = tid; i < n; i += stride) {
        float v = x[i];
        float sp = softplus_stable(v);
        float t = tanhf(sp);
        y[i] = v * t;
    }
}

torch::Tensor fused_act_hip(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA/ROCm tensor");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto y = torch::empty_like(x);
    int64_t n64 = x.numel();
    TORCH_CHECK(n64 <= INT_MAX, "numel too large");
    int n = (int)n64;

    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 4096) blocks = 4096;

    hipLaunchKernelGGL(
        fused_act_kernel,
        dim3(blocks),
        dim3(threads),
        0,
        at::cuda::getDefaultCUDAStream(),
        (const float*)x.data_ptr<float>(),
        (float*)y.data_ptr<float>(),
        n
    );
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_act_hip", &fused_act_hip, "fused activation: x*tanh(softplus(x)) (ROCm)");
}
'''

fused_act_ext = load_inline(
    name='fused_act_ext',
    cpp_sources='',
    cuda_sources=hip_src,
    functions=None,
    extra_cuda_cflags=['-O3'],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """Conv2d + fused activation (custom HIP) + BatchNorm2d."""
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self._ext = fused_act_ext

    def forward(self, x):
        x = self.conv(x)
        x = self._ext.fused_act_hip(x.contiguous())
        x = self.bn(x)
        return x


def get_inputs():
    batch_size = 64
    in_channels = 64
    height, width = 128, 128
    return [torch.rand(batch_size, in_channels, height, width, device='cuda', dtype=torch.float32)]


def get_init_inputs():
    in_channels = 64
    out_channels = 128
    kernel_size = 3
    return [in_channels, out_channels, kernel_size]
