
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

gelu_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define INV_SQRT_2 0.70710678118654752440f

__device__ __forceinline__ float gelu_func(float x) {
    return 0.5f * x * (1.0f + erff(x * INV_SQRT_2));
}

__global__ void gelu_kernel_unrolled(const float* __restrict__ input, float* __restrict__ output, int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    
    if (idx + 15 < size) {
        float4 in_v0 = reinterpret_cast<const float4*>(&input[idx])[0];
        float4 in_v1 = reinterpret_cast<const float4*>(&input[idx + 4])[0];
        float4 in_v2 = reinterpret_cast<const float4*>(&input[idx + 8])[0];
        float4 in_v3 = reinterpret_cast<const float4*>(&input[idx + 12])[0];
        
        float4 out_v0, out_v1, out_v2, out_v3;
        out_v0.x = gelu_func(in_v0.x);
        out_v0.y = gelu_func(in_v0.y);
        out_v0.z = gelu_func(in_v0.z);
        out_v0.w = gelu_func(in_v0.w);
        
        out_v1.x = gelu_func(in_v1.x);
        out_v1.y = gelu_func(in_v1.y);
        out_v1.z = gelu_func(in_v1.z);
        out_v1.w = gelu_func(in_v1.w);
        
        out_v2.x = gelu_func(in_v2.x);
        out_v2.y = gelu_func(in_v2.y);
        out_v2.z = gelu_func(in_v2.z);
        out_v2.w = gelu_func(in_v2.w);
        
        out_v3.x = gelu_func(in_v3.x);
        out_v3.y = gelu_func(in_v3.y);
        out_v3.z = gelu_func(in_v3.z);
        out_v3.w = gelu_func(in_v3.w);
        
        reinterpret_cast<float4*>(&output[idx])[0] = out_v0;
        reinterpret_cast<float4*>(&output[idx + 4])[0] = out_v1;
        reinterpret_cast<float4*>(&output[idx + 8])[0] = out_v2;
        reinterpret_cast<float4*>(&output[idx + 12])[0] = out_v3;
    } else {
        for (int i = idx; i < size; ++i) {
            output[i] = gelu_func(input[i]);
        }
    }
}

torch::Tensor gelu_hip(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    
    const int block_size = 256;
    const int num_blocks = (size / 16 + block_size - 1) / block_size;
    
    gelu_kernel_unrolled<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    return output;
}
"""

gelu_lib = load_inline(
    name="gelu_lib",
    cpp_sources=gelu_source,
    functions=["gelu_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu_lib = gelu_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu_lib.gelu_hip(x)
