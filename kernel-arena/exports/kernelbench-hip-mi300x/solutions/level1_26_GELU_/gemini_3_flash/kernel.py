
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

gelu_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__global__ void gelu_kernel_v16(const float* __restrict__ input, float* __restrict__ output, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < n) {
        float4 in_vec = reinterpret_cast<const float4*>(input)[idx / 4];
        float4 out_vec;
        
        out_vec.x = 0.5f * in_vec.x * (1.0f + erff(in_vec.x * 0.7071067811865475f));
        out_vec.y = 0.5f * in_vec.y * (1.0f + erff(in_vec.y * 0.7071067811865475f));
        out_vec.z = 0.5f * in_vec.z * (1.0f + erff(in_vec.z * 0.7071067811865475f));
        out_vec.w = 0.5f * in_vec.w * (1.0f + erff(in_vec.w * 0.7071067811865475f));
        
        reinterpret_cast<float4*>(output)[idx / 4] = out_vec;
    } else {
        for (int i = idx; i < n; ++i) {
            float x = input[i];
            output[i] = 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
        }
    }
}

torch::Tensor gelu_hip(torch::Tensor input) {
    auto n = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 1024;
    const int num_blocks = (n / 4 + block_size - 1) / block_size;

    gelu_kernel_v16<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), n);

    return output;
}
"""

gelu_cpp_source = """
torch::Tensor gelu_hip(torch::Tensor input);
"""

gelu_lib = load_inline(
    name="gelu_lib",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu_lib = gelu_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu_lib.gelu_hip(x)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []
