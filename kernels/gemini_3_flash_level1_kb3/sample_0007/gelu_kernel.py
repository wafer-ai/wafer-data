
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

gelu_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__device__ __forceinline__ float gelu_func(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

__global__ void gelu_kernel_final(const float* __restrict__ input, float* __restrict__ output, int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    
    if (idx + 7 < size) {
        float4 in1 = reinterpret_cast<const float4*>(input)[idx / 4];
        float4 in2 = reinterpret_cast<const float4*>(input)[idx / 4 + 1];
        
        float4 out1, out2;
        
        out1.x = gelu_func(in1.x);
        out1.y = gelu_func(in1.y);
        out1.z = gelu_func(in1.z);
        out1.w = gelu_func(in1.w);
        
        out2.x = gelu_func(in2.x);
        out2.y = gelu_func(in2.y);
        out2.z = gelu_func(in2.z);
        out2.w = gelu_func(in2.w);
        
        reinterpret_cast<float4*>(output)[idx / 4] = out1;
        reinterpret_cast<float4*>(output)[idx / 4 + 1] = out2;
    } else {
        for (int i = idx; i < size; ++i) {
            output[i] = gelu_func(input[i]);
        }
    }
}

torch::Tensor gelu_hip(torch::Tensor x) {
    auto output = torch::empty_like(x);
    int size = x.numel();
    
    const int block_size = 256;
    const int num_blocks = (size + (block_size * 8) - 1) / (block_size * 8);
    
    gelu_kernel_final<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        output.data_ptr<float>(), 
        size
    );
    
    return output;
}
"""

gelu_cpp_source = "torch::Tensor gelu_hip(torch::Tensor x);"

gelu_module = load_inline(
    name="gelu_hip_v5",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_kernel_source,
    functions=["gelu_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu_module = gelu_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            return torch.nn.functional.gelu(x)
        return self.gelu_module.gelu_hip(x)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []
