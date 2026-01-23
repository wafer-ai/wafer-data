
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

elu_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__device__ __forceinline__ float elu_func(float val, float alpha) {
    return (val > 0.0f) ? val : alpha * (expm1f(val));
}

__global__ void elu_kernel_vectorized(const float* __restrict__ x, float* __restrict__ out, float alpha, int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < size) {
        float4 in_v = *reinterpret_cast<const float4*>(&x[idx]);
        float4 out_v;
        out_v.x = elu_func(in_v.x, alpha);
        out_v.y = elu_func(in_v.y, alpha);
        out_v.z = elu_func(in_v.z, alpha);
        out_v.w = elu_func(in_v.w, alpha);
        *reinterpret_cast<float4*>(&out[idx]) = out_v;
    } else {
        for (int i = idx; i < size; ++i) {
            out[i] = elu_func(x[i], alpha);
        }
    }
}

torch::Tensor elu_hip(torch::Tensor x, float alpha) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 512;
    const int num_blocks = (size + block_size * 4 - 1) / (block_size * 4);

    elu_kernel_vectorized<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, size);

    return out;
}
"""

elu_cpp_source = "torch::Tensor elu_hip(torch::Tensor x, float alpha);"

elu_op = load_inline(
    name="elu_op",
    cpp_sources=elu_cpp_source,
    cuda_sources=elu_kernel_source,
    functions=["elu_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super(ModelNew, self).__init__()
        self.alpha = float(alpha)
        self.elu_op = elu_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and x.dtype == torch.float32:
            return self.elu_op.elu_hip(x, self.alpha)
        else:
            return torch.nn.functional.elu(x, alpha=self.alpha)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return [1.0]
