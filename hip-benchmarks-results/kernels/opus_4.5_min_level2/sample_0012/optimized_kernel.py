import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused divide + GELU kernel
fused_div_gelu_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <cmath>

__device__ __forceinline__ float gelu_approx(float x) {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // Using tanh approximation for speed
    const float kSqrt2OverPi = 0.7978845608028654f;  // sqrt(2/pi)
    const float kCoeff = 0.044715f;
    float x3 = x * x * x;
    float inner = kSqrt2OverPi * (x + kCoeff * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float gelu_exact(float x) {
    // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    const float kSqrt2Inv = 0.7071067811865475f;  // 1/sqrt(2)
    return x * 0.5f * (1.0f + erff(x * kSqrt2Inv));
}

__global__ void fused_div_gelu_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       const float divisor_inv,
                                       const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process 4 elements per thread for better memory coalescing
    int idx4 = idx * 4;
    
    if (idx4 + 3 < size) {
        // Load 4 floats at once using float4
        float4 in_val = *reinterpret_cast<const float4*>(input + idx4);
        
        // Apply division and GELU
        float x0 = in_val.x * divisor_inv;
        float x1 = in_val.y * divisor_inv;
        float x2 = in_val.z * divisor_inv;
        float x3 = in_val.w * divisor_inv;
        
        float4 out_val;
        out_val.x = gelu_exact(x0);
        out_val.y = gelu_exact(x1);
        out_val.z = gelu_exact(x2);
        out_val.w = gelu_exact(x3);
        
        *reinterpret_cast<float4*>(output + idx4) = out_val;
    } else {
        // Handle remaining elements
        for (int i = 0; i < 4 && idx4 + i < size; i++) {
            float x = input[idx4 + i] * divisor_inv;
            output[idx4 + i] = gelu_exact(x);
        }
    }
}

torch::Tensor fused_div_gelu_hip(torch::Tensor input, float divisor) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    // Each thread handles 4 elements
    const int num_blocks = (size + block_size * 4 - 1) / (block_size * 4);
    
    float divisor_inv = 1.0f / divisor;
    
    fused_div_gelu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        divisor_inv,
        size
    );
    
    return output;
}
"""

fused_div_gelu_cpp = """
torch::Tensor fused_div_gelu_hip(torch::Tensor input, float divisor);
"""

fused_div_gelu = load_inline(
    name="fused_div_gelu",
    cpp_sources=fused_div_gelu_cpp,
    cuda_sources=fused_div_gelu_source,
    functions=["fused_div_gelu_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "--offload-arch=gfx942"]
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses divide and GELU into a single kernel.
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor
        self.fused_div_gelu = fused_div_gelu

    def forward(self, x):
        # Use PyTorch's optimized linear (rocBLAS)
        x = self.linear(x)
        # Use our fused division + GELU kernel
        x = self.fused_div_gelu.fused_div_gelu_hip(x, self.divisor)
        return x


batch_size = 1024
input_size = 8192
output_size = 8192
divisor = 10.0


def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]


def get_init_inputs():
    return [input_size, output_size, divisor]
