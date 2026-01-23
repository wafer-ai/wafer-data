
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

tanh_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__global__ void __launch_bounds__(256) tanh_unroll_kernel(const float4* __restrict__ x, float4* __restrict__ out, int size_v4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size_v4) {
        float4 val = x[idx];
        val.x = tanhf(val.x);
        val.y = tanhf(val.y);
        val.z = tanhf(val.z);
        val.w = tanhf(val.w);
        out[idx] = val;
    }
}

torch::Tensor tanh_hip(torch::Tensor x) {
    auto out = torch::empty_like(x);
    int size = x.numel();
    const int block_size = 256;
    int size_v4 = size / 4;
    const int num_blocks = (size_v4 + block_size - 1) / block_size;
    
    tanh_unroll_kernel<<<num_blocks, block_size>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(out.data_ptr<float>()),
        size_v4
    );
    return out;
}
"""

tanh_cpp_source = "torch::Tensor tanh_hip(torch::Tensor x);"

tanh_module = load_inline(
    name="tanh_module_best",
    cpp_sources=tanh_cpp_source,
    cuda_sources=tanh_source,
    functions=["tanh_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tanh_hip = tanh_module.tanh_hip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and x.numel() % 4 == 0:
            return self.tanh_hip(x)
        else:
            return torch.tanh(x)
