import os
os.environ["CXX"] = "hipcc"
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

gelu_cpp = """
#include &lt;hip/hip_runtime.h&gt;
#include &lt;torch/extension.h&gt;
#include &lt;cmath&gt;

__global__ void gelu_kernel(const float *input, float *output, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx &lt; n) {
        float x = input[idx];
        float x2 = x * x;
        float x3 = x2 * x;
        float d = 0.044715f * x3 + x;
        d *= 0.797885f;
        float tanh_d = tanhf(d);
        float res = 0.5f * (1.0f + tanh_d);
        output[idx] = x * res;
    }
}

torch::Tensor custom_gelu(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat, "input must be float32");
    auto output = torch::empty_like(input);
    size_t n = input.numel();
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (n &gt; 0) {
        hipLaunchKernelGGL(gelu_kernel, dim3(blocks), dim3(threads), 0, 0, input.data_ptr&lt;float&gt;(), output.data_ptr&lt;float&gt;(), n);
    }
    return output;
}
"""

gelu_module = load_inline(name="gelu", cpp_sources=gelu_cpp, functions=["custom_gelu"], verbose=True)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gelu = gelu_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu.custom_gelu(x)
