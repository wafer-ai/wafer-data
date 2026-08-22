import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Ensure we compile with HIPCC on ROCm
os.environ.setdefault("CXX", "hipcc")
os.environ.setdefault("CC", "hipcc")

# Fused Swish (SiLU) + scaling, in-place on FP32.
# This replaces: x = x * sigmoid(x); x = x * scaling_factor
swish_scale_cpp_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

__device__ __forceinline__ float sigmoidf_fast(float x) {
    // Fast sigmoid using expf; HIP provides device expf.
    return 1.0f / (1.0f + expf(-x));
}

__global__ void swish_scale_inplace_vec4_kernel(float* __restrict__ x, int64_t n, float scale) {
    int64_t base = (int64_t)(blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (base + 3 < n) {
        float4 v = *reinterpret_cast<const float4*>(x + base);
        float s0 = sigmoidf_fast(v.x);
        float s1 = sigmoidf_fast(v.y);
        float s2 = sigmoidf_fast(v.z);
        float s3 = sigmoidf_fast(v.w);
        v.x = v.x * s0 * scale;
        v.y = v.y * s1 * scale;
        v.z = v.z * s2 * scale;
        v.w = v.w * s3 * scale;
        *reinterpret_cast<float4*>(x + base) = v;
    } else {
        // Tail (including cases where n < 4)
        for (int64_t i = base; i < n && i < base + 4; ++i) {
            float v = x[i];
            float s = sigmoidf_fast(v);
            x[i] = v * s * scale;
        }
    }
}

torch::Tensor swish_scale_inplace_hip(torch::Tensor x, double scaling_factor) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA/HIP tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be FP32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto n = x.numel();
    if (n == 0) return x;

    const int threads = 256;
    const int64_t vec_elems = (n + 3) / 4; // number of vec4 work-items
    const int blocks = (int)((vec_elems + threads - 1) / threads);

    hipStream_t stream = (hipStream_t)at::cuda::getDefaultCUDAStream();
    swish_scale_inplace_vec4_kernel<<<blocks, threads, 0, stream>>>(
        (float*)x.data_ptr<float>(), n, (float)scaling_factor);

    return x;
}
"""

swish_scale_ext = load_inline(
    name="swish_scale_ext",
    cpp_sources=swish_scale_cpp_source,
    functions=["swish_scale_inplace_hip"],
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """Optimized version: keep rocBLAS GEMM for Linear, fuse Swish+scaling into one in-place HIP kernel."""

    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)
        self._swish_scale = swish_scale_ext

    def forward(self, x):
        x = self.matmul(x)
        # In-place fused activation + scaling
        return self._swish_scale.swish_scale_inplace_hip(x, self.scaling_factor)


# Keep the same shapes / helpers as the reference
batch_size = 128
in_features = 32768
out_features = 32768
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]
