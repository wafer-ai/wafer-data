import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

gelu_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Fast GELU approximation using tanh
// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
// This is PyTorch's default GELU approximation

__device__ __forceinline__ float gelu_exact(float x) {
    const float sqrt_2_inv = 0.7071067811865475f;  // 1/sqrt(2)
    return x * 0.5f * (1.0f + erff(x * sqrt_2_inv));
}

// Main kernel with aggressive vectorization and ILP
__global__ __launch_bounds__(512)
void gelu_kernel_vec4_fast(const float4* __restrict__ input, 
                            float4* __restrict__ output, 
                            int n_vec4) {
    const float sqrt_2_inv = 0.7071067811865475f;
    
    // Each thread processes multiple float4s for better ILP
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int idx = tid; idx < n_vec4; idx += stride) {
        float4 in_vec = input[idx];
        
        float4 out_vec;
        out_vec.x = in_vec.x * 0.5f * (1.0f + erff(in_vec.x * sqrt_2_inv));
        out_vec.y = in_vec.y * 0.5f * (1.0f + erff(in_vec.y * sqrt_2_inv));
        out_vec.z = in_vec.z * 0.5f * (1.0f + erff(in_vec.z * sqrt_2_inv));
        out_vec.w = in_vec.w * 0.5f * (1.0f + erff(in_vec.w * sqrt_2_inv));
        
        output[idx] = out_vec;
    }
}

// Handle remaining elements
__global__ void gelu_kernel_remainder(const float* __restrict__ input, 
                                       float* __restrict__ output, 
                                       int start, int n) {
    const float sqrt_2_inv = 0.7071067811865475f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    
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
    
    const int block_size = 512;
    
    // Process most elements with vectorized kernel
    int n_vec4 = n / 4;
    int vec_elements = n_vec4 * 4;
    
    if (n_vec4 > 0) {
        // Use grid-stride loop pattern - launch enough blocks to saturate the GPU
        int num_blocks = min((n_vec4 + block_size - 1) / block_size, 2048);
        
        gelu_kernel_vec4_fast<<<num_blocks, block_size>>>(
            reinterpret_cast<const float4*>(input.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            n_vec4
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
    extra_cuda_cflags=["-O3", "-ffast-math", "--gpu-max-threads-per-block=512"],
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
