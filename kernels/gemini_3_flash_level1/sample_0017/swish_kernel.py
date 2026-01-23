
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Set HIP compiler
os.environ["CXX"] = "hipcc"

swish_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__device__ inline float swish_func(float x) {
    // Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
    return x / (1.0f + __expf(-x));
}

__global__ void swish_kernel_vec4(const float* __restrict__ x, float* __restrict__ out, int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < size) {
        float4 in_vec = reinterpret_cast<const float4*>(x)[idx / 4];
        float4 out_vec;
        out_vec.x = swish_func(in_vec.x);
        out_vec.y = swish_func(in_vec.y);
        out_vec.z = swish_func(in_vec.z);
        out_vec.w = swish_func(in_vec.w);
        reinterpret_cast<float4*>(out)[idx / 4] = out_vec;
    } else {
        for (int i = idx; i < size; ++i) {
            out[i] = swish_func(x[i]);
        }
    }
}

torch::Tensor swish_hip(torch::Tensor x) {
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    auto out = torch::empty_like(x);
    int size = x.numel();
    
    const int block_size = 256;
    int num_threads = (size + 3) / 4;
    const int num_blocks = (num_threads + block_size - 1) / block_size;
    
    hipLaunchKernelGGL(swish_kernel_vec4, dim3(num_blocks), dim3(block_size), 0, 0, 
                       x.data_ptr<float>(), out.data_ptr<float>(), size);
    
    return out;
}
"""

swish_lib = load_inline(
    name="swish_hip_vec4_final",
    cpp_sources=swish_cpp_source,
    functions=["swish_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.swish_lib = swish_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            return x * torch.sigmoid(x)
        return self.swish_lib.swish_hip(x)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []
