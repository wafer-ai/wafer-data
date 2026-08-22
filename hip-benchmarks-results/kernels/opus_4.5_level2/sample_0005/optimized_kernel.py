import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused Mish activation kernel: x * tanh(softplus(x))
# softplus(x) = log(1 + exp(x))
# We use numerically stable version to avoid overflow
mish_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

__device__ __forceinline__ float mish_activation(float x) {
    // Numerically stable softplus: log(1 + exp(x))
    // For large x, softplus(x) ≈ x
    // For small x, use log(1 + exp(x))
    float sp;
    if (x > 20.0f) {
        sp = x;
    } else if (x < -20.0f) {
        sp = expf(x);
    } else {
        sp = log1pf(expf(x));
    }
    return x * tanhf(sp);
}

__global__ void mish_kernel(const float* __restrict__ input, 
                            float* __restrict__ output,
                            int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process 4 elements per thread for better memory coalescing
    int idx4 = idx * 4;
    
    if (idx4 + 3 < size) {
        float4 in_val = *reinterpret_cast<const float4*>(input + idx4);
        float4 out_val;
        out_val.x = mish_activation(in_val.x);
        out_val.y = mish_activation(in_val.y);
        out_val.z = mish_activation(in_val.z);
        out_val.w = mish_activation(in_val.w);
        *reinterpret_cast<float4*>(output + idx4) = out_val;
    } else if (idx4 < size) {
        // Handle remaining elements
        for (int i = idx4; i < size && i < idx4 + 4; i++) {
            output[i] = mish_activation(input[i]);
        }
    }
}

torch::Tensor mish_hip(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = ((size + 3) / 4 + block_size - 1) / block_size;
    
    mish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        size
    );
    
    return output;
}
"""

mish_cpp_source = """
torch::Tensor mish_hip(torch::Tensor input);
"""

mish_module = load_inline(
    name="mish_activation",
    cpp_sources=mish_cpp_source,
    cuda_sources=mish_kernel_source,
    functions=["mish_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model with fused Mish activation kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.mish = mish_module

    def forward(self, x):
        x = self.conv(x)
        # Fused mish activation: x * tanh(softplus(x))
        x = self.mish.mish_hip(x)
        x = self.bn(x)
        return x


def get_inputs():
    return [torch.rand(64, 64, 128, 128).cuda()]


def get_init_inputs():
    return [64, 128, 3]
