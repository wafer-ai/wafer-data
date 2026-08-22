import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

gelu_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Simple GELU kernel - each thread handles one element
__global__ void gelu_kernel_simple(const float* __restrict__ input, 
                                    float* __restrict__ output, 
                                    int n) {
    const float sqrt_2_inv = 0.7071067811865475f;  // 1/sqrt(2)
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float x = input[idx];
        output[idx] = x * 0.5f * (1.0f + erff(x * sqrt_2_inv));
    }
}

torch::Tensor gelu_hip(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
    
    auto output = torch::empty_like(input);
    int n = input.numel();
    
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;
    
    gelu_kernel_simple<<<num_blocks, block_size>>>(
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
