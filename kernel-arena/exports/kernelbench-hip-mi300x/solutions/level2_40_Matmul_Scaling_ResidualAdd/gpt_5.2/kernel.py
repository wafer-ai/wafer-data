import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Ensure we compile with hipcc on ROCm
os.environ.setdefault("CXX", "hipcc")

# Fused kernel: replaces clone().detach() + mul + add with a single in-place scale.
# Reference computes:
#   y = Linear(x)
#   original_y = y.clone().detach()
#   y = y * scaling_factor
#   out = y + original_y
# Forward-value equivalence:
#   out = (1 + scaling_factor) * y
# So we can do y *= (1 + scaling_factor) directly, eliminating the huge clone.

_hip_src = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <c10/cuda/CUDAStream.h>

namespace {

__global__ void scale_inplace_f32_kernel(float* __restrict__ x, float alpha, int64_t n) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t idx4 = tid * 4;

    if (idx4 + 3 < n) {
        float4 v = *reinterpret_cast<const float4*>(x + idx4);
        v.x *= alpha;
        v.y *= alpha;
        v.z *= alpha;
        v.w *= alpha;
        *reinterpret_cast<float4*>(x + idx4) = v;
    } else {
        for (int64_t i = idx4; i < n; ++i) {
            x[i] *= alpha;
        }
    }
}

} // namespace

torch::Tensor scale_inplace_f32_hip(torch::Tensor x, double alpha) {
    TORCH_CHECK(x.is_cuda(), "scale_inplace_f32_hip: expected CUDA/HIP tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "scale_inplace_f32_hip: expected float32 tensor");
    TORCH_CHECK(x.is_contiguous(), "scale_inplace_f32_hip: expected contiguous tensor");

    const auto n = x.numel();
    if (n == 0) return x;

    const int threads = 256;
    // Each thread handles 4 elements (float4) except tail
    const int64_t blocks = (n + (threads * 4) - 1) / (threads * 4);

    auto stream = c10::cuda::getCurrentCUDAStream();
    hipStream_t hip_stream = (hipStream_t)stream.stream();

    scale_inplace_f32_kernel<<<(dim3)blocks, (dim3)threads, 0, hip_stream>>>(
        (float*)x.data_ptr<float>(), (float)alpha, n);

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scale_inplace_f32_hip", &scale_inplace_f32_hip, "In-place FP32 scale (HIP)");
}
'''

_scale_ext = load_inline(
    name="kb_scale_inplace_f32_hip_ext",
    cpp_sources=_hip_src,
    functions=None,
    extra_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)
        self._alpha = 1.0 + self.scaling_factor

    def forward(self, x):
        y = self.matmul(x)
        # Fused (clone+mul+add) -> one in-place scale.
        return _scale_ext.scale_inplace_f32_hip(y, self._alpha)


batch_size = 16384
in_features = 4096
out_features = 4096
scaling_factor = 0.5

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]
