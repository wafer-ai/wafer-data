import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Compile HIP extension (ROCm)
os.environ.setdefault("CXX", "hipcc")
os.environ.setdefault("CC", "hipcc")

# Fused: (x / divisor) -> GELU
# We keep nn.Linear (rocBLAS/hipBLAS) for the GEMM, and fuse the divide+GELU into one kernel.

hip_src = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>

// GELU(x) = 0.5*x*(1 + erf(x/sqrt(2)))
__device__ __forceinline__ float gelu_erf(float x) {
    const float inv_sqrt2 = 0.70710678118654752440f;
    return 0.5f * x * (1.0f + erff(x * inv_sqrt2));
}

__global__ void div_gelu_kernel_vec4(const float4* __restrict__ in4,
                                    float4* __restrict__ out4,
                                    int64_t n4,
                                    float inv_div) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < n4; i += stride) {
        float4 v = in4[i];
        v.x = gelu_erf(v.x * inv_div);
        v.y = gelu_erf(v.y * inv_div);
        v.z = gelu_erf(v.z * inv_div);
        v.w = gelu_erf(v.w * inv_div);
        out4[i] = v;
    }
}

__global__ void div_gelu_kernel(const float* __restrict__ in,
                               float* __restrict__ out,
                               int64_t n,
                               float inv_div) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        out[i] = gelu_erf(in[i] * inv_div);
    }
}

torch::Tensor div_gelu_hip(torch::Tensor x, double divisor) {
    TORCH_CHECK(x.is_cuda(), "div_gelu_hip: expected a CUDA/HIP tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "div_gelu_hip: expected FP32 tensor");
    TORCH_CHECK(x.is_contiguous(), "div_gelu_hip: expected contiguous tensor");

    auto out = torch::empty_like(x);
    const int64_t n = x.numel();
    const float inv_div = (float)(1.0 / divisor);

    // Heuristic launch params
    const int threads = 256;
    // Use a moderate number of blocks; this is memory-bound.
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 4096) blocks = 4096;

    const uintptr_t in_ptr = (uintptr_t)x.data_ptr<float>();
    const uintptr_t out_ptr = (uintptr_t)out.data_ptr<float>();

    if ((n % 4 == 0) && (in_ptr % 16 == 0) && (out_ptr % 16 == 0)) {
        const int64_t n4 = n / 4;
        hipLaunchKernelGGL(div_gelu_kernel_vec4, dim3(blocks), dim3(threads), 0, 0,
                           (const float4*)x.data_ptr<float>(),
                           (float4*)out.data_ptr<float>(),
                           n4,
                           inv_div);
    } else {
        hipLaunchKernelGGL(div_gelu_kernel, dim3(blocks), dim3(threads), 0, 0,
                           x.data_ptr<float>(),
                           out.data_ptr<float>(),
                           n,
                           inv_div);
    }

    return out;
}
"""

# Build extension lazily/cached
_div_gelu_ext = load_inline(
    name="div_gelu_ext_86",
    cpp_sources=hip_src,
    functions=["div_gelu_hip"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """Optimized: keep Linear GEMM, fuse divide+GELU into one HIP kernel."""

    def __init__(self, input_size, output_size, divisor):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = float(divisor)

    def forward(self, x):
        x = self.linear(x)
        return _div_gelu_ext.div_gelu_hip(x, self.divisor)


# Keep the same benchmark harness API
batch_size = 1024
input_size = 8192
output_size = 8192
divisor = 10.0

def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, output_size, divisor]
