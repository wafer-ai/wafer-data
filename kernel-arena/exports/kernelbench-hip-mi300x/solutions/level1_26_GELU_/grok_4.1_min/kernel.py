import os
os.environ["CXX"] = "hipcc"

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

gelu_cpp = """
#include <hip/hip_runtime.h>
#include <hip_vector_types.h>
#include <cmath>

__global__ void gelu_kernel(const float *g_i, float *g_o, size_t num_floats) {
    size_t idx4 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_float4 = num_floats / 4;
    if (idx4 < num_float4) {
        const float4* input = reinterpret_cast<const float4 *>(g_i);
        float4* output = reinterpret_cast<float4 *>(g_o);
        float4 x4 = input[idx4];
        const float kErfApproxB1 = 0.707106781186547524400844362104849f;
        float e0 = erf(x4.x * kErfApproxB1);
        float e1 = erf(x4.y * kErfApproxB1);
        float e2 = erf(x4.z * kErfApproxB1);
        float e3 = erf(x4.w * kErfApproxB1);
        float4 o4;
        o4.x = 0.5f * x4.x * (1.0f + e0);
        o4.y = 0.5f * x4.y * (1.0f + e1);
        o4.z = 0.5f * x4.z * (1.0f + e2);
        o4.w = 0.5f * x4.w * (1.0f + e3);
        output[idx4] = o4;
    }
}

torch::Tensor gelu_hip(torch::Tensor input) {
    torch::Tensor output = torch::empty_like(input);
    int64_t n = input.numel();
    if (n == 0) {
        return output;
    }
    const int block_size = 256;
    size_t num_float4 = static_cast<size_t>(n) / 4;
    int grid_size = (num_float4 + block_size - 1) / block_size;
    gelu_kernel<<<grid_size, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), static_cast<size_t>(n));
    return output;
}
"""

gelu = load_inline(
    name="gelu",
    cpp_sources=gelu_cpp,
    functions=["gelu_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gelu = gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu.gelu_hip(x)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
