import os
os.environ["CXX"] = "hipcc"
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

gelu_cpp = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>
#include <algorithm>

__global__ void gelu_kernel(const float *input, float *output, int64_t n) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t gid = tid; gid < (size_t)n; gid += stride) {
        float x = input[gid];
        float x2 = x * x;
        float x3 = x2 * x;
        float d = 0.044715f * x3 + x;
        d *= 0.7978845608f;
        float t = tanhf(d);
        output[gid] = 0.5f * x * (1.0f + t);
    }
}

torch::Tensor custom_gelu_hip(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int64_t n = input.numel();
    const int block_size = 256;
    int full_blocks = (n + block_size - 1LL) / block_size;
    int num_blocks = std::min(131072, full_blocks);
    gelu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}
"""

gelu_module = load_inline(
    name="gelu",
    cpp_sources=gelu_cpp,
    functions=["custom_gelu_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gelu = gelu_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu.custom_gelu_hip(x)
