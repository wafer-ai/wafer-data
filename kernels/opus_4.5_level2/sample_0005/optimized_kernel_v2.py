import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized Mish activation kernel with better vectorization
mish_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

// Compute mish: x * tanh(softplus(x))
// Using fast approximations for better performance
__device__ __forceinline__ float fast_mish(float x) {
    // Numerically stable softplus with fast math
    float sp;
    if (x > 20.0f) {
        sp = x;
    } else if (x < -20.0f) {
        sp = __expf(x);  // Fast exp
    } else {
        sp = __logf(1.0f + __expf(x));  // Fast log and exp
    }
    return x * tanhf(sp);
}

// Kernel using float4 vectorization for coalesced memory access
__global__ void mish_kernel_vectorized(const float* __restrict__ input, 
                                        float* __restrict__ output,
                                        int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // Each thread processes 4 floats at a time
    int vec_size = size / 4;
    
    for (int i = tid; i < vec_size; i += stride) {
        float4 in_val = reinterpret_cast<const float4*>(input)[i];
        float4 out_val;
        out_val.x = fast_mish(in_val.x);
        out_val.y = fast_mish(in_val.y);
        out_val.z = fast_mish(in_val.z);
        out_val.w = fast_mish(in_val.w);
        reinterpret_cast<float4*>(output)[i] = out_val;
    }
    
    // Handle remainder
    int base = vec_size * 4;
    for (int i = base + tid; i < size; i += stride) {
        output[i] = fast_mish(input[i]);
    }
}

torch::Tensor mish_hip(torch::Tensor input) {
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 512;
    // Use more blocks for better occupancy on MI300X
    const int num_blocks = std::min(65535, (size / 4 + block_size - 1) / block_size);
    
    mish_kernel_vectorized<<<num_blocks, block_size>>>(
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
    name="mish_activation_v2",
    cpp_sources=mish_cpp_source,
    cuda_sources=mish_kernel_source,
    functions=["mish_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math", "--save-temps"]
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
