import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Ensure we build with hipcc on ROCm
os.environ.setdefault("CXX", "hipcc")

# Exact GELU: 0.5*x*(1+erf(x/sqrt(2)))
# Optimizations:
#  - Vectorized float4 loads/stores when contiguous
#  - Each thread processes 4 elements per iteration
#  - Use erff (device) for FP32

cpp_source = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>

#ifndef __HIP_PLATFORM_HCC__
#define __HIP_PLATFORM_HCC__
#endif

static inline __device__ float gelu_exact_f32(float x) {
    // 0.5 * x * (1 + erf(x / sqrt(2)))
    const float inv_sqrt2 = 0.70710678118654752440f;
    return 0.5f * x * (1.0f + erff(x * inv_sqrt2));
}

__global__ void gelu_f32_vec4_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t idx4 = tid * 4;
    int64_t stride4 = (int64_t)gridDim.x * blockDim.x * 4;

    for (int64_t i = idx4; i < n; i += stride4) {
        // Process up to 4 elements
        float v0, v1, v2, v3;
        if (i + 3 < n) {
            // Vectorized load
            float4 v = *reinterpret_cast<const float4*>(x + i);
            v0 = v.x; v1 = v.y; v2 = v.z; v3 = v.w;

            float4 o;
            o.x = gelu_exact_f32(v0);
            o.y = gelu_exact_f32(v1);
            o.z = gelu_exact_f32(v2);
            o.w = gelu_exact_f32(v3);
            *reinterpret_cast<float4*>(y + i) = o;
        } else {
            // Tail
            if (i < n) y[i] = gelu_exact_f32(x[i]);
            if (i + 1 < n) y[i + 1] = gelu_exact_f32(x[i + 1]);
            if (i + 2 < n) y[i + 2] = gelu_exact_f32(x[i + 2]);
        }
    }
}

torch::Tensor gelu_exact_hip(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA/HIP tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto y = torch::empty_like(x);
    int64_t n = x.numel();

    // Heuristic launch: enough blocks to cover MI300X
    const int threads = 256;
    int blocks = (int)((n + (threads * 4) - 1) / (threads * 4));
    // Clamp to a reasonable max to avoid huge grid on very large tensors
    if (blocks > 65535) blocks = 65535;

    hipLaunchKernelGGL(gelu_f32_vec4_kernel,
                       dim3(blocks), dim3(threads), 0, 0,
                       (const float*)x.data_ptr<float>(), (float*)y.data_ptr<float>(), n);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gelu_exact_hip", &gelu_exact_hip, "GELU exact (FP32) HIP");
}
'''

# Build extension
_gelu_ext = load_inline(
    name="gelu_exact_ext",
    cpp_sources=cpp_source,
    functions=None,
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback to PyTorch if not on GPU/FP32/contiguous
        if (not x.is_cuda) or x.dtype != torch.float32 or (not x.is_contiguous()):
            return torch.nn.functional.gelu(x)
        return _gelu_ext.gelu_exact_hip(x)


# KernelBench helpers
batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]

def get_init_inputs():
    return []
