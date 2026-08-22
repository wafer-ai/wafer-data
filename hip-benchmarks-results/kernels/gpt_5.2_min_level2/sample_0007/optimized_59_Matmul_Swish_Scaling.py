import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ.setdefault("CXX", "hipcc")

cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

// Fast sigmoid using fast exp; acceptable for FP32 benchmarking.
__device__ __forceinline__ float sigmoid_fast(float x) {
    // 1 / (1 + exp(-x))
    return 1.0f / (1.0f + __expf(-x));
}

__global__ void swish_scale_inplace_kernel(float* __restrict__ x, int64_t n, float scale) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    // Vectorized if 16B aligned
    uintptr_t addr = (uintptr_t)x;
    if ((addr & 0xF) == 0) {
        int64_t n4 = n / 4;
        float4* x4 = reinterpret_cast<float4*>(x);
        for (int64_t i = tid; i < n4; i += stride) {
            float4 v = x4[i];
            float s0 = sigmoid_fast(v.x);
            float s1 = sigmoid_fast(v.y);
            float s2 = sigmoid_fast(v.z);
            float s3 = sigmoid_fast(v.w);
            v.x = (v.x * s0) * scale;
            v.y = (v.y * s1) * scale;
            v.z = (v.z * s2) * scale;
            v.w = (v.w * s3) * scale;
            x4[i] = v;
        }
        int64_t start = n4 * 4;
        for (int64_t i = start + tid; i < n; i += stride) {
            float v = x[i];
            x[i] = (v * sigmoid_fast(v)) * scale;
        }
    } else {
        for (int64_t i = tid; i < n; i += stride) {
            float v = x[i];
            x[i] = (v * sigmoid_fast(v)) * scale;
        }
    }
}

torch::Tensor swish_scale_inplace_hip(torch::Tensor x, double scale) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA/ROCm tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be FP32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int64_t n = x.numel();
    if (n == 0) return x;

    const int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    blocks = blocks > 4096 ? 4096 : blocks;

    hipStream_t stream = (hipStream_t)at::cuda::getDefaultCUDAStream();
    swish_scale_inplace_kernel<<<blocks, threads, 0, stream>>>(
        (float*)x.data_ptr<float>(), n, (float)scale
    );
    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swish_scale_inplace_hip", &swish_scale_inplace_hip, "Inplace fused swish+scale (FP32, HIP)");
}
"""

swish_scale_ext = load_inline(
    name="swish_scale_ext_59_v2",
    cpp_sources="",
    cuda_sources=cuda_src,
    functions=None,
    extra_cuda_cflags=["-O3", "-ffast-math"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """Optimized: keep GEMM in nn.Linear; do Swish+Scaling in-place in one HIP kernel."""

    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        x = self.matmul(x)
        return swish_scale_ext.swish_scale_inplace_hip(x, self.scaling_factor)


batch_size = 128
in_features = 32768
out_features = 32768
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features, device="cuda", dtype=torch.float32)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]
