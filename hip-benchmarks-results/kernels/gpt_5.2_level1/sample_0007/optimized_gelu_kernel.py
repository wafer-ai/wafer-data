import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ.setdefault("CXX", "hipcc")

# Vectorized FP32 GELU (exact, erf-based) for ROCm/HIP.
# Key optimization vs a generic elementwise kernel: float4 vectorized IO +
# launching on PyTorch's *current* HIP stream (avoids stream mismatch sync).
source = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/HIPStream.h>

static inline __device__ __forceinline__ float gelu_exact_f32(float x) {
    const float inv_sqrt2 = 0.70710678118654752440f;
    return 0.5f * x * (1.0f + erff(x * inv_sqrt2));
}

__global__ void gelu_vec4_kernel(const float4* __restrict__ x4,
                                float4* __restrict__ y4,
                                int n4) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int stride = (int)(blockDim.x * gridDim.x);

    for (int i = tid; i < n4; i += stride) {
        float4 v = x4[i];
        v.x = gelu_exact_f32(v.x);
        v.y = gelu_exact_f32(v.y);
        v.z = gelu_exact_f32(v.z);
        v.w = gelu_exact_f32(v.w);
        y4[i] = v;
    }
}

__global__ void gelu_tail_kernel(const float* __restrict__ x,
                                float* __restrict__ y,
                                int start,
                                int n) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x) + start;
    if (idx < n) {
        y[idx] = gelu_exact_f32(x[idx]);
    }
}

torch::Tensor gelu_hip(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "gelu_hip: expected CUDA/HIP tensor");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, "gelu_hip: expected FP32 tensor");
    TORCH_CHECK(x.is_contiguous(), "gelu_hip: expected contiguous tensor");

    auto y = torch::empty_like(x);

    const int64_t n = x.numel();
    if (n == 0) return y;

    const int threads = 256;
    const int64_t n4 = n / 4;
    const int64_t tail = n - n4 * 4;

    hipStream_t stream = c10::hip::getCurrentHIPStream();

    if (n4 > 0) {
        // Keep enough blocks to fill the GPU; avoid overlaunching.
        // MI300X has lots of CUs; 8192 blocks is ample.
        int blocks = (int)((n4 + threads - 1) / threads);
        if (blocks > 8192) blocks = 8192;

        const float4* x4 = reinterpret_cast<const float4*>(x.data_ptr<float>());
        float4* y4 = reinterpret_cast<float4*>(y.data_ptr<float>());
        hipLaunchKernelGGL(gelu_vec4_kernel, dim3(blocks), dim3(threads), 0, stream, x4, y4, (int)n4);
    }

    if (tail) {
        int start = (int)(n4 * 4);
        int blocks_tail = (int)((tail + threads - 1) / threads);
        hipLaunchKernelGGL(gelu_tail_kernel, dim3(blocks_tail), dim3(threads), 0, stream,
                           x.data_ptr<float>(), y.data_ptr<float>(), start, (int)n);
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gelu_hip", &gelu_hip, "FP32 GELU (HIP)");
}
"""

_gelu_ext = load_inline(
    name="gelu_hip_ext",
    cpp_sources=source,
    functions=None,
    with_cuda=True,
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _gelu_ext.gelu_hip(x)


batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
