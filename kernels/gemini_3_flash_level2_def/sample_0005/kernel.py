
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

mish_kernel_code = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__device__ __forceinline__ float fast_mish(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return x * expf(x);
    float e_x = expf(x);
    float e_s = 1.0f + e_x;
    float e_s2 = e_s * e_s;
    return x * (e_s2 - 1.0f) / (e_s2 + 1.0f);
}

__global__ void mish_kernel_vec(const float* __restrict__ input, float* __restrict__ output, int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < size) {
        float4 in_v = reinterpret_cast<const float4*>(input + idx)[0];
        float4 out_v;
        out_v.x = fast_mish(in_v.x);
        out_v.y = fast_mish(in_v.y);
        out_v.z = fast_mish(in_v.z);
        out_v.w = fast_mish(in_v.w);
        reinterpret_cast<float4*>(output + idx)[0] = out_v;
    } else {
        for (int i = idx; i < size; ++i) {
            output[i] = fast_mish(input[i]);
        }
    }
}

torch::Tensor mish_hip(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int block_size = 256;
    const int num_blocks = (size / 4 + block_size - 1) / block_size;
    
    mish_kernel_vec<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}
"""

mish_lib = load_inline(
    name="mish_lib",
    cpp_sources=mish_kernel_code,
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
