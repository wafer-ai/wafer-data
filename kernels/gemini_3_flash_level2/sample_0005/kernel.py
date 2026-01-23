
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

mish_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__device__ inline float mish_func(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return x * expf(x);
    float ep1 = 1.0f + expf(x);
    return x * (1.0f - 2.0f / (1.0f + ep1 * ep1));
}

__global__ void mish_kernel_vec4(float* data, int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < size) {
        float4 val = reinterpret_cast<float4*>(&data[idx])[0];
        val.x = mish_func(val.x);
        val.y = mish_func(val.y);
        val.z = mish_func(val.z);
        val.w = mish_func(val.w);
        reinterpret_cast<float4*>(&data[idx])[0] = val;
    } else {
        for (int i = idx; i < size; ++i) {
            data[i] = mish_func(data[i]);
        }
    }
}

torch::Tensor mish_hip(torch::Tensor x) {
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    auto size = x.numel();
    const int block_size = 256;
    const int num_blocks = (size / 4 + block_size - 1) / block_size;
    mish_kernel_vec4<<<num_blocks, block_size>>>(x.data_ptr<float>(), size);
    return x;
}
"""

mish_lib = load_inline(
    name="mish_lib_v4",
    cpp_sources=mish_source,
    functions=["mish_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.conv(x)
        x = mish_lib.mish_hip(x)
        x = self.bn(x)
        return x

def get_inputs():
    batch_size = 64
    in_channels = 64
    height, width = 128, 128
    return [torch.randn(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    in_channels = 64
    out_channels = 128
    kernel_size = 3
    return [in_channels, out_channels, kernel_size]
