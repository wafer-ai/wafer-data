import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Use HIP compiler for ROCm
os.environ.setdefault("CXX", "hipcc")

# Optimization:
# The reference does:
#   y = Linear(x)
#   original = y.clone().detach()
#   y = y * scaling_factor
#   y = y + original
# which is equivalent to:
#   y = Linear(x) * (1 + scaling_factor)
# This removes a massive device-to-device clone and an extra elementwise add.

scale_cpp = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>

template <int VEC>
__global__ void scale_kernel(const float* __restrict__ inp, float* __restrict__ out, float factor, int n) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int idx = tid * VEC;
    if (idx >= n) return;

    #pragma unroll
    for (int i = 0; i < VEC; i++) {
        int j = idx + i;
        if (j < n) out[j] = inp[j] * factor;
    }
}

torch::Tensor scale_fp32(torch::Tensor x, double factor_d) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA/HIP tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");

    auto out = torch::empty_like(x);
    const int n = (int)x.numel();
    const float factor = (float)factor_d;

    uintptr_t addr_in = (uintptr_t)x.data_ptr<float>();
    uintptr_t addr_out = (uintptr_t)out.data_ptr<float>();
    bool aligned16 = ((addr_in | addr_out) & 0xF) == 0;

    const int threads = 256;
    if (aligned16) {
        const int VEC = 4;
        int blocks = (n + (threads * VEC - 1)) / (threads * VEC);
        hipLaunchKernelGGL((scale_kernel<VEC>), dim3(blocks), dim3(threads), 0, 0,
                           (const float*)x.data_ptr<float>(), (float*)out.data_ptr<float>(), factor, n);
    } else {
        const int VEC = 1;
        int blocks = (n + (threads * VEC - 1)) / (threads * VEC);
        hipLaunchKernelGGL((scale_kernel<VEC>), dim3(blocks), dim3(threads), 0, 0,
                           (const float*)x.data_ptr<float>(), (float*)out.data_ptr<float>(), factor, n);
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scale_fp32", &scale_fp32, "Scale fp32 tensor (HIP)");
}
"""

scale_ext = load_inline(
    name="scale_ext_fp32",
    cpp_sources=scale_cpp,
    functions=None,
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.register_buffer("_fused_factor", torch.tensor(1.0 + float(scaling_factor), dtype=torch.float32))

    def forward(self, x):
        y = self.matmul(x)
        return scale_ext.scale_fp32(y, float(self._fused_factor.item()))


def get_inputs():
    batch_size = 16384
    in_features = 4096
    return [torch.rand(batch_size, in_features, device="cuda", dtype=torch.float32)]


def get_init_inputs():
    in_features = 4096
    out_features = 4096
    scaling_factor = 0.5
    return [in_features, out_features, scaling_factor]
