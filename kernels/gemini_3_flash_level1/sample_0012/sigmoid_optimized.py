
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

sigmoid_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__global__ void sigmoid_kernel_vec4(const float4* __restrict__ x, float4* __restrict__ out, int size_vec4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size_vec4) {
        float4 val = x[idx];
        float4 res;
        res.x = sigmoid_f(val.x);
        res.y = sigmoid_f(val.y);
        res.z = sigmoid_f(val.z);
        res.w = sigmoid_f(val.w);
        out[idx] = res;
    }
}

__global__ void sigmoid_kernel_scalar(const float* __restrict__ x, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = sigmoid_f(x[idx]);
    }
}

torch::Tensor sigmoid_hip(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;

    if (size % 4 == 0) {
        int size_vec4 = size / 4;
        const int num_blocks = (size_vec4 + block_size - 1) / block_size;
        sigmoid_kernel_vec4<<<num_blocks, block_size>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            size_vec4
        );
    } else {
        const int num_blocks = (size + block_size - 1) / block_size;
        sigmoid_kernel_scalar<<<num_blocks, block_size>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            size
        );
    }

    return out;
}
"""

sigmoid_module = load_inline(
    name="sigmoid_module_final",
    cpp_sources=sigmoid_cpp_source,
    functions=["sigmoid_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.sigmoid_hip = sigmoid_module.sigmoid_hip
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid_hip(x)

def get_inputs():
    batch_size = 4096
    dim = 393216
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []
