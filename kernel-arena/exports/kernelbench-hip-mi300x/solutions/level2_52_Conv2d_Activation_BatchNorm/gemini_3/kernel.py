
import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

mish_source = """
#include <hip/hip_runtime.h>

__device__ __forceinline__ float mish_op(float x) {
    // Optimization: Algebraic simplification of tanh(softplus(x))
    // softplus(x) = log(1 + exp(x))
    // tanh(y) = (exp(2y) - 1)/(exp(2y) + 1)
    // Substituting y = log(1 + exp(x)):
    // tanh(softplus(x)) = ( (1+e^x)^2 - 1 ) / ( (1+e^x)^2 + 1 )
    //                   = ( 2e^x + e^2x ) / ( 2 + 2e^x + e^2x )
    // Let e = exp(x).
    // result = x * ( e*(2+e) / (2 + e*(2+e)) )
    
    // Stability check:
    // If x > 20, exp(x) is large, softplus(x) approx x, tanh(x) approx 1. Mish(x) approx x.
    // We use 20.0f threshold to avoid overflow in expf and match PyTorch softplus threshold.
    if (x > 20.0f) return x;
    
    float e = expf(x);
    float n = e * (2.0f + e);
    return x * (n / (2.0f + n));
}

__global__ void mish_kernel_vec(const float* __restrict__ inp, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int vec_size = size / 4;
    
    const float4* inp_v = reinterpret_cast<const float4*>(inp);
    float4* out_v = reinterpret_cast<float4*>(out);
    
    for (int i = idx; i < vec_size; i += stride) {
        float4 v = inp_v[i];
        v.x = mish_op(v.x);
        v.y = mish_op(v.y);
        v.z = mish_op(v.z);
        v.w = mish_op(v.w);
        out_v[i] = v;
    }
    
    int start_rem = vec_size * 4;
    for (int i = start_rem + idx; i < size; i += stride) {
        out[i] = mish_op(inp[i]);
    }
}

__global__ void mish_kernel_scalar(const float* __restrict__ inp, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += stride) {
        out[i] = mish_op(inp[i]);
    }
}

torch::Tensor mish_hip(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int block_size = 256;
    
    // Check alignment for float4
    bool aligned = (reinterpret_cast<uintptr_t>(input.data_ptr<float>()) % 16 == 0) &&
                   (reinterpret_cast<uintptr_t>(output.data_ptr<float>()) % 16 == 0);

    if (aligned && (size % 4 == 0)) {
        int vec_elements = size / 4;
        int num_blocks = std::min(65535, (vec_elements + block_size - 1) / block_size);
        mish_kernel_vec<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    } else {
        int num_blocks = std::min(65535, (size + block_size - 1) / block_size);
        mish_kernel_scalar<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    }
    
    return output;
}
"""

mish_module = load_inline(
    name="mish_module_opt",
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
        x = mish_module.mish_hip(x)
        x = self.bn(x)
        return x

def get_inputs():
    batch_size = 64
    in_channels = 64
    height, width = 128, 128
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [64, 128, 3]
