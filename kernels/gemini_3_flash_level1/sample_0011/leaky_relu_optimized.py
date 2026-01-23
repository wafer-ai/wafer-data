
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

leaky_relu_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void leaky_relu_kernel_vectorized(const float* __restrict__ x, float* __restrict__ out, int size, float negative_slope) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < size) {
        float4 val = reinterpret_cast<const float4*>(x + idx)[0];
        val.x = (val.x >= 0) ? val.x : val.x * negative_slope;
        val.y = (val.y >= 0) ? val.y : val.y * negative_slope;
        val.z = (val.z >= 0) ? val.z : val.z * negative_slope;
        val.w = (val.w >= 0) ? val.w : val.w * negative_slope;
        reinterpret_cast<float4*>(out + idx)[0] = val;
    } else {
        for (int i = idx; i < size; ++i) {
            float val = x[i];
            out[i] = (val >= 0) ? val : val * negative_slope;
        }
    }
}

torch::Tensor leaky_relu_hip(torch::Tensor x, float negative_slope) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size / 4 + block_size - 1) / block_size;

    hipLaunchKernelGGL(leaky_relu_kernel_vectorized, dim3(num_blocks), dim3(block_size), 0, 0,
                       x.data_ptr<float>(), out.data_ptr<float>(), (int)size, negative_slope);

    return out;
}
"""

leaky_relu_module = load_inline(
    name="leaky_relu_hip",
    cpp_sources=leaky_relu_source,
    functions=["leaky_relu_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope
        self.leaky_relu_hip = leaky_relu_module.leaky_relu_hip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            return torch.nn.functional.leaky_relu(x, negative_slope=self.negative_slope)
        return self.leaky_relu_hip(x, self.negative_slope)
