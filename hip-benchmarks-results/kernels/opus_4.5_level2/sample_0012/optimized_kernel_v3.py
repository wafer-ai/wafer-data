import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized fused divide + GELU kernel with large blocks and better memory access
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Fast GELU approximation using tanh
__device__ __forceinline__ float fast_gelu(float x) {
    const float c1 = 0.7978845608028654f; // sqrt(2/pi)
    const float c2 = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(c1 * (x + c2 * x3)));
}

// Warp-level optimized kernel
__global__ __launch_bounds__(512) void fused_div_gelu_kernel_opt(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float inv_divisor,
    const int size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_stride = blockDim.x * gridDim.x;
    
    // Each thread processes 4 elements per iteration with grid stride
    for (int base = tid * 4; base < size; base += grid_stride * 4) {
        if (base + 3 < size) {
            // Aligned vector load
            float4 data = *reinterpret_cast<const float4*>(&input[base]);
            
            // Fused divide + GELU
            data.x = fast_gelu(data.x * inv_divisor);
            data.y = fast_gelu(data.y * inv_divisor);
            data.z = fast_gelu(data.z * inv_divisor);
            data.w = fast_gelu(data.w * inv_divisor);
            
            // Aligned vector store
            *reinterpret_cast<float4*>(&output[base]) = data;
        } else {
            // Handle tail elements
            for (int i = base; i < size; ++i) {
                output[i] = fast_gelu(input[i] * inv_divisor);
            }
        }
    }
}

torch::Tensor fused_div_gelu_hip(torch::Tensor input, float divisor) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
    
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const float inv_divisor = 1.0f / divisor;
    
    // Use large block size and moderate grid size for occupancy
    const int block_size = 512;
    const int elements_per_thread = 4;
    const int threads_needed = (size + elements_per_thread - 1) / elements_per_thread;
    const int num_blocks = std::min((threads_needed + block_size - 1) / block_size, 2048);
    
    hipLaunchKernelGGL(fused_div_gelu_kernel_opt, dim3(num_blocks), dim3(block_size), 0, 0,
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        inv_divisor,
        size
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_div_gelu_hip(torch::Tensor input, float divisor);
"""

fused_module = load_inline(
    name="fused_div_gelu_opt",
    cpp_sources=cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_div_gelu_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math", "--offload-arch=gfx942"]
)


class ModelNew(nn.Module):
    """
    Optimized model with fused divide + GELU.
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor

    def forward(self, x):
        # Use PyTorch's optimized linear layer (uses rocBLAS)
        x = self.linear(x)
        # Use fused divide + GELU kernel
        x = fused_module.fused_div_gelu_hip(x, self.divisor)
        return x
