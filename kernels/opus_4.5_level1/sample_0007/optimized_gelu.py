import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

gelu_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
// Using the exact GELU formula with erf

__device__ __forceinline__ float gelu_scalar(float x) {
    const float sqrt_2_inv = 0.7071067811865475f;  // 1/sqrt(2)
    return x * 0.5f * (1.0f + erff(x * sqrt_2_inv));
}

// Vectorized GELU kernel using float4 for coalesced memory access
__global__ void gelu_kernel_vec4(const float* __restrict__ input, 
                                  float* __restrict__ output, 
                                  int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 4;
    
    if (vec_idx + 3 < n) {
        // Load 4 floats at once
        float4 in_vec = *reinterpret_cast<const float4*>(input + vec_idx);
        
        float4 out_vec;
        out_vec.x = gelu_scalar(in_vec.x);
        out_vec.y = gelu_scalar(in_vec.y);
        out_vec.z = gelu_scalar(in_vec.z);
        out_vec.w = gelu_scalar(in_vec.w);
        
        // Store 4 floats at once
        *reinterpret_cast<float4*>(output + vec_idx) = out_vec;
    }
}

// Handle remaining elements
__global__ void gelu_kernel_remainder(const float* __restrict__ input, 
                                       float* __restrict__ output, 
                                       int start, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    
    if (idx < n) {
        output[idx] = gelu_scalar(input[idx]);
    }
}

torch::Tensor gelu_hip(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
    
    auto output = torch::empty_like(input);
    int n = input.numel();
    
    const int block_size = 256;
    
    // Process most elements with vectorized kernel
    int vec_elements = (n / 4) * 4;  // Round down to multiple of 4
    int num_vec_blocks = (vec_elements / 4 + block_size - 1) / block_size;
    
    if (num_vec_blocks > 0) {
        gelu_kernel_vec4<<<num_vec_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            n
        );
    }
    
    // Handle remaining elements
    int remainder = n - vec_elements;
    if (remainder > 0) {
        int num_rem_blocks = (remainder + block_size - 1) / block_size;
        gelu_kernel_remainder<<<num_rem_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            vec_elements,
            n
        );
    }
    
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
