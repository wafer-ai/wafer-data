import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

gelu_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// GELU approximation using tanh (faster than erf)
// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
// However, PyTorch uses exact GELU by default, so we must use erf

// Vectorized GELU kernel
__global__ void gelu_kernel_vec4(const float* __restrict__ input, 
                                  float* __restrict__ output, 
                                  int n) {
    const float sqrt_2_inv = 0.7071067811865475f;
    
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < n) {
        // Manual vectorization with float4 load
        float4 in_val = *reinterpret_cast<const float4*>(&input[idx]);
        
        float4 out_val;
        out_val.x = in_val.x * 0.5f * (1.0f + erff(in_val.x * sqrt_2_inv));
        out_val.y = in_val.y * 0.5f * (1.0f + erff(in_val.y * sqrt_2_inv));
        out_val.z = in_val.z * 0.5f * (1.0f + erff(in_val.z * sqrt_2_inv));
        out_val.w = in_val.w * 0.5f * (1.0f + erff(in_val.w * sqrt_2_inv));
        
        *reinterpret_cast<float4*>(&output[idx]) = out_val;
    } else if (idx < n) {
        // Handle tail
        for (int i = idx; i < n && i < idx + 4; i++) {
            float x = input[i];
            output[i] = x * 0.5f * (1.0f + erff(x * sqrt_2_inv));
        }
    }
}

torch::Tensor gelu_hip(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
    
    auto output = torch::empty_like(input);
    int n = input.numel();
    
    const int block_size = 1024;
    const int num_elements_per_block = block_size * 4;
    const int num_blocks = (n + num_elements_per_block - 1) / num_elements_per_block;
    
    gelu_kernel_vec4<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output;
}
"""

gelu_cpp_source = """
torch::Tensor gelu_hip(torch::Tensor input);
"""

gelu_module = load_inline(
    name="gelu_hip",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_hip_source,
    functions=["gelu_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu_op = gelu_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu_op.gelu_hip(x)


def get_inputs():
    x = torch.rand(4096, 393216).cuda()
    return [x]


def get_init_inputs():
    return []
