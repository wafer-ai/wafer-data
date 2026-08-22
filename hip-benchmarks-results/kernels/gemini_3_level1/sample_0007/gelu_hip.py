import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>
#include <cmath>

#define SQRT_1_2 0.70710678118654752440f

__global__ void __launch_bounds__(1024) gelu_kernel_vec4_1024(const float* __restrict__ x, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_size = size / 4;
    
    if (idx < vec_size) {
        const float4* x_ptr = reinterpret_cast<const float4*>(x);
        float4* out_ptr = reinterpret_cast<float4*>(out);
        
        float4 v = x_ptr[idx];
        float4 r;
        
        r.x = 0.5f * v.x * (1.0f + erff(v.x * SQRT_1_2));
        r.y = 0.5f * v.y * (1.0f + erff(v.y * SQRT_1_2));
        r.z = 0.5f * v.z * (1.0f + erff(v.z * SQRT_1_2));
        r.w = 0.5f * v.w * (1.0f + erff(v.w * SQRT_1_2));
        
        out_ptr[idx] = r;
    }
}

__global__ void gelu_kernel_scalar(const float* __restrict__ x, float* __restrict__ out, int size, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < size) {
        float v = x[idx];
        out[idx] = 0.5f * v * (1.0f + erff(v * SQRT_1_2));
    }
}

torch::Tensor gelu_hip(torch::Tensor x) {
    // x is assumed contiguous from generator
    
    auto size = x.numel();
    auto out = torch::empty_like(x);
    
    int vec_size = size / 4;
    int remainder = size % 4;
    
    const int block_size = 1024;
    
    if (vec_size > 0) {
        int num_blocks = (vec_size + block_size - 1) / block_size;
        gelu_kernel_vec4_1024<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
    }
    
    if (remainder > 0) {
        int offset = vec_size * 4;
        gelu_kernel_scalar<<<1, remainder>>>(x.data_ptr<float>(), out.data_ptr<float>(), size, offset);
    }
    
    return out;
}
"""

gelu_module = load_inline(
    name="gelu_module_v5",
    cpp_sources=cpp_source,
    functions=["gelu_hip"],
    extra_cflags=['-O3'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu_module = gelu_module
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu_module.gelu_hip(x)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
