import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

gelu_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Process multiple float4s per thread with register-based prefetch
__global__ __launch_bounds__(256)
void gelu_kernel_opt(const float4* __restrict__ input, 
                      float4* __restrict__ output, 
                      const int n_vec4) {
    const float sqrt_2_inv = 0.7071067811865475f;
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    
    // Each thread processes multiple float4s with a stride pattern
    #pragma unroll 4
    for (int idx = tid; idx < n_vec4; idx += total_threads) {
        float4 in_val = __builtin_nontemporal_load(&input[idx]);
        
        float4 out_val;
        out_val.x = in_val.x * 0.5f * (1.0f + erff(in_val.x * sqrt_2_inv));
        out_val.y = in_val.y * 0.5f * (1.0f + erff(in_val.y * sqrt_2_inv));
        out_val.z = in_val.z * 0.5f * (1.0f + erff(in_val.z * sqrt_2_inv));
        out_val.w = in_val.w * 0.5f * (1.0f + erff(in_val.w * sqrt_2_inv));
        
        __builtin_nontemporal_store(out_val, &output[idx]);
    }
}

// Scalar kernel for the tail
__global__ void gelu_kernel_tail(const float* __restrict__ input, 
                                  float* __restrict__ output, 
                                  const int start, const int n) {
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
    
    int n_vec4 = n / 4;
    int vec_elements = n_vec4 * 4;
    
    if (n_vec4 > 0) {
        const int block_size = 256;
        // Launch enough blocks to fully utilize the GPU
        const int num_blocks = min((n_vec4 + block_size - 1) / block_size, 65535);
        
        gelu_kernel_opt<<<num_blocks, block_size>>>(
            reinterpret_cast<const float4*>(input.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            n_vec4
        );
    }
    
    // Handle remaining elements
    int remainder = n - vec_elements;
    if (remainder > 0) {
        const int block_size = 256;
        int num_rem_blocks = (remainder + block_size - 1) / block_size;
        gelu_kernel_tail<<<num_rem_blocks, block_size>>>(
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
    name="gelu_hip_v5",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_hip_source,
    functions=["gelu_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math", "-munsafe-fp-atomics"],
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
