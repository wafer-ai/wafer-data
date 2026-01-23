
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

relu_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void relu_kernel_vec4(const float4* __restrict__ x, float4* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float4 val = x[idx];
        val.x = fmaxf(0.0f, val.x);
        val.y = fmaxf(0.0f, val.y);
        val.z = fmaxf(0.0f, val.z);
        val.w = fmaxf(0.0f, val.w);
        out[idx] = val;
    }
}

torch::Tensor relu_hip(torch::Tensor x) {
    auto out = torch::empty_like(x);
    int total_elements = x.numel();
    
    int vec_size = total_elements / 4;
    const int block_size = 256;
    const int num_blocks = (vec_size + block_size - 1) / block_size;
    relu_kernel_vec4<<<num_blocks, block_size>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()), 
        reinterpret_cast<float4*>(out.data_ptr<float>()), 
        vec_size
    );
    
    return out;
}
"""

relu_module = load_inline(
    name="relu_v6",
    cpp_sources=relu_cpp_source,
    functions=["relu_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.relu_hip = relu_module.relu_hip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu_hip(x)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []
