
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

swish_scaling_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void swish_scaling_kernel_vec(float* x, float scaling_factor, int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx < size) {
        float4 val4 = reinterpret_cast<float4*>(&x[idx])[0];
        
        val4.x = (val4.x / (1.0f + __expf(-val4.x))) * scaling_factor;
        val4.y = (val4.y / (1.0f + __expf(-val4.y))) * scaling_factor;
        val4.z = (val4.z / (1.0f + __expf(-val4.z))) * scaling_factor;
        val4.w = (val4.w / (1.0f + __expf(-val4.w))) * scaling_factor;
        
        reinterpret_cast<float4*>(&x[idx])[0] = val4;
    }
}

void swish_scaling_hip(torch::Tensor x, float scaling_factor) {
    int size = x.numel();
    const int block_size = 256;
    const int num_blocks = (size / 4 + block_size - 1) / block_size;

    swish_scaling_kernel_vec<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        scaling_factor,
        size
    );
}
"""

swish_scaling_lib = load_inline(
    name="swish_scaling_final",
    cpp_sources=swish_scaling_source,
    functions=["swish_scaling_hip"],
    verbose=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        # Use F.linear directly for the best performance
        x = F.linear(x, self.matmul.weight, self.matmul.bias)
        # Apply the fused elementwise activation and scaling in-place
        swish_scaling_lib.swish_scaling_hip(x, self.scaling_factor)
        return x

def get_inputs():
    batch_size = 128
    in_features = 32768
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    in_features = 32768
    out_features = 32768
    scaling_factor = 2.0
    return [in_features, out_features, scaling_factor]
