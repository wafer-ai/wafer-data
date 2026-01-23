
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

fused_ops_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__device__ __forceinline__ float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

__global__ void div_gelu_tanh_kernel_vec8(float* out, float inv_divisor, int total_elements) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (idx + 7 < total_elements) {
        float4 v1 = reinterpret_cast<float4*>(&out[idx])[0];
        float4 v2 = reinterpret_cast<float4*>(&out[idx + 4])[0];
        
        v1.x = gelu_tanh(v1.x * inv_divisor);
        v1.y = gelu_tanh(v1.y * inv_divisor);
        v1.z = gelu_tanh(v1.z * inv_divisor);
        v1.w = gelu_tanh(v1.w * inv_divisor);
        
        v2.x = gelu_tanh(v2.x * inv_divisor);
        v2.y = gelu_tanh(v2.y * inv_divisor);
        v2.z = gelu_tanh(v2.z * inv_divisor);
        v2.w = gelu_tanh(v2.w * inv_divisor);
        
        reinterpret_cast<float4*>(&out[idx])[0] = v1;
        reinterpret_cast<float4*>(&out[idx + 4])[0] = v2;
    } else {
        for (int i = idx; i < total_elements; ++i) {
            out[i] = gelu_tanh(out[i] * inv_divisor);
        }
    }
}

void div_gelu_tanh(torch::Tensor out, float inv_divisor) {
    int total_elements = out.numel();
    const int block_size = 256;
    const int num_blocks = (total_elements / 8 + block_size - 1) / block_size;
    div_gelu_tanh_kernel_vec8<<<num_blocks, block_size>>>(out.data_ptr<float>(), inv_divisor, total_elements);
}
"""

fused_ops = load_inline(
    name="fused_ops_v5",
    cpp_sources=fused_ops_source,
    functions=["div_gelu_tanh"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor
        self.fused_ops = fused_ops

    def forward(self, x):
        # Use F.linear to fuse bias addition. It's usually the fastest way.
        out = torch.nn.functional.linear(x, self.linear.weight, self.linear.bias)
        # Apply fused division and tanh-approximated GELU
        self.fused_ops.div_gelu_tanh(out, 1.0 / float(self.divisor))
        return out

def get_inputs():
    batch_size = 1024
    input_size = 8192
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    input_size = 8192
    output_size = 8192
    divisor = 10.0
    return [input_size, output_size, divisor]
