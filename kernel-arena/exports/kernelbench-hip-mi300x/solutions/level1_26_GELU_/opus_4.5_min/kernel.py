import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

gelu_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Use exact GELU: GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
__device__ __forceinline__ float gelu_exact(float x) {
    const float inv_sqrt2 = 0.7071067811865475f;
    return x * 0.5f * (1.0f + erff(x * inv_sqrt2));
}

// Grid-stride vectorized GELU kernel using float4
__global__ void gelu_kernel_vec4(const float4* __restrict__ input, 
                                  float4* __restrict__ output,
                                  int n_vec4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop for handling large arrays
    for (; idx < n_vec4; idx += grid_stride) {
        float4 in_val = __ldg(&input[idx]);  // Use cached load
        float4 out_val;
        out_val.x = gelu_exact(in_val.x);
        out_val.y = gelu_exact(in_val.y);
        out_val.z = gelu_exact(in_val.z);
        out_val.w = gelu_exact(in_val.w);
        output[idx] = out_val;
    }
}

// Scalar kernel for remaining elements (won't be called for this problem)
__global__ void gelu_kernel_scalar(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int start_idx,
                                    int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + start_idx;
    if (idx < total_size) {
        output[idx] = gelu_exact(input[idx]);
    }
}

torch::Tensor gelu_hip(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
    
    auto output = torch::empty_like(input);
    int total_size = input.numel();
    
    // Use vectorized kernel for bulk of data
    int n_vec4 = total_size / 4;
    int remainder = total_size % 4;
    
    // MI300X has 304 CUs. Use 1024 threads per block for max occupancy.
    const int block_size = 1024;
    // Use enough blocks to saturate the GPU
    const int max_blocks = 65535;
    
    if (n_vec4 > 0) {
        int num_blocks = (n_vec4 + block_size - 1) / block_size;
        if (num_blocks > max_blocks) num_blocks = max_blocks;
        
        gelu_kernel_vec4<<<num_blocks, block_size>>>(
            reinterpret_cast<const float4*>(input.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            n_vec4
        );
    }
    
    // Handle remaining elements with scalar kernel
    if (remainder > 0) {
        int start_idx = n_vec4 * 4;
        int num_blocks_scalar = (remainder + block_size - 1) / block_size;
        gelu_kernel_scalar<<<num_blocks_scalar, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            start_idx,
            total_size
        );
    }
    
    return output;
}
"""

gelu_cpp_source = """
torch::Tensor gelu_hip(torch::Tensor input);
"""

gelu_module = load_inline(
    name="gelu_hip_v4",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_hip_source,
    functions=["gelu_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs GELU activation using custom HIP kernel.
    """
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
