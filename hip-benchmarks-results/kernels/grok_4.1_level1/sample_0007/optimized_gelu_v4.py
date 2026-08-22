import os
os.environ["CXX"] = "hipcc"
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

gelu_cpp = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__global__ void gelu_kernel(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float x2 = x * x;
        float x3 = x2 * x;
        float d = 0.044715f * x3 + x;
        d *= 0.79788456f;
        float t = tanhf(d);
        output[idx] = 0.5f * x * (1.0f + t);
    }
}

torch::Tensor custom_gelu_hip(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int n = input.numel();
    const int block_size = 1024;
    const int num_blocks = (n + block_size - 1) / block_size;
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
