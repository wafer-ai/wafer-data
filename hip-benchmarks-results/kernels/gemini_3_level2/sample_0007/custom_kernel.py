import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>
#include <cmath>

__global__ void fused_swish_scale_inplace_kernel(float* __restrict__ data, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    int vec_size = size / 4;
    float4* data_vec = reinterpret_cast<float4*>(data);
    
    for (int i = idx; i < vec_size; i += stride) {
        float4 v = data_vec[i];
        float4 r;
        
        // Fast swish calculation
        // x * (1 / (1 + exp(-x))) * scale
        // Using __expf for potentially faster exp
        
        r.x = v.x * (1.0f / (1.0f + __expf(-v.x))) * scale;
        r.y = v.y * (1.0f / (1.0f + __expf(-v.y))) * scale;
        r.z = v.z * (1.0f / (1.0f + __expf(-v.z))) * scale;
        r.w = v.w * (1.0f / (1.0f + __expf(-v.w))) * scale;
        
        data_vec[i] = r;
    }
    
    int rem_start = vec_size * 4;
    for (int i = rem_start + idx; i < size; i += stride) {
        float v = data[i];
        data[i] = v * (1.0f / (1.0f + __expf(-v))) * scale;
    }
}

void fused_swish_scale_inplace(torch::Tensor x, float scale) {
    int size = x.numel();
    
    const int block_size = 256;
    int vec_size = size / 4;
    int num_blocks = std::min(65535, (vec_size + block_size - 1) / block_size);
    if (num_blocks == 0) num_blocks = 1;
    
    fused_swish_scale_inplace_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), scale, size);
}
"""

module = load_inline(
    name="fused_swish_scale_inplace_impl",
    cpp_sources=cpp_source,
    functions=["fused_swish_scale_inplace"],
    verbose=False,
    extra_cflags=['-O3']
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.matmul(x)
        module.fused_swish_scale_inplace(x, self.scaling_factor)
        return x

batch_size = 128
in_features = 32768
out_features = 32768
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]
