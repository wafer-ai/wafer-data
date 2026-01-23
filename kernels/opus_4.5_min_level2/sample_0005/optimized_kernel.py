import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused Mish activation kernel: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
mish_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

__global__ void mish_kernel(const float* __restrict__ input, 
                            float* __restrict__ output, 
                            int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = input[idx];
        // softplus(x) = ln(1 + exp(x))
        // For numerical stability, use: softplus(x) = x + ln(1 + exp(-|x|)) if x > 0
        //                               softplus(x) = ln(1 + exp(x)) if x <= 0
        float sp;
        if (x > 20.0f) {
            sp = x;  // For large x, softplus(x) ≈ x
        } else if (x < -20.0f) {
            sp = expf(x);  // For very negative x, softplus(x) ≈ exp(x)
        } else {
            sp = log1pf(expf(x));
        }
        // mish = x * tanh(softplus(x))
        float tanh_sp = tanhf(sp);
        output[idx] = x * tanh_sp;
    }
}

// Vectorized version using float4 for better memory throughput
__global__ void mish_kernel_vec4(const float4* __restrict__ input, 
                                  float4* __restrict__ output, 
                                  int size4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size4) {
        float4 in = input[idx];
        float4 out;
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float x = (i == 0) ? in.x : (i == 1) ? in.y : (i == 2) ? in.z : in.w;
            float sp;
            if (x > 20.0f) {
                sp = x;
            } else if (x < -20.0f) {
                sp = expf(x);
            } else {
                sp = log1pf(expf(x));
            }
            float result = x * tanhf(sp);
            if (i == 0) out.x = result;
            else if (i == 1) out.y = result;
            else if (i == 2) out.z = result;
            else out.w = result;
        }
        
        output[idx] = out;
    }
}

torch::Tensor mish_hip(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    
    // Use vectorized version if size is divisible by 4 and aligned
    if (size % 4 == 0 && ((uintptr_t)input.data_ptr<float>() % 16 == 0)) {
        int size4 = size / 4;
        int num_blocks = (size4 + block_size - 1) / block_size;
        mish_kernel_vec4<<<num_blocks, block_size>>>(
            reinterpret_cast<const float4*>(input.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            size4
        );
    } else {
        int num_blocks = (size + block_size - 1) / block_size;
        mish_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            size
        );
    }
    
    return output;
}
"""

mish_cpp_source = """
torch::Tensor mish_hip(torch::Tensor input);
"""

mish_module = load_inline(
    name="mish_module",
    cpp_sources=mish_cpp_source,
    cuda_sources=mish_kernel_source,
    functions=["mish_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "--offload-arch=gfx942"]
)


class ModelNew(nn.Module):
    """
    Optimized model with fused Mish activation kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.conv(x)
        # Fused Mish activation: x * tanh(softplus(x))
        x = mish_module.mish_hip(x)
        x = self.bn(x)
        return x
