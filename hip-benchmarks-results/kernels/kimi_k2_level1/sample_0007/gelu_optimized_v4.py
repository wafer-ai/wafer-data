import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

gelu_cpp_source = """
#include <hip/hip_runtime.h>
#include <math.h>

#define SQRT_2_OVER_PI 0.7978845608028654f
#define COEFF 0.044715f

__device__ __forceinline__ float gelu_scalar(float x) {
    // Fast GELU computation using tanh approximation
    float x3 = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
    float tanh_val = tanhf(inner);
    return 0.5f * x * (1.0f + tanh_val);
}

__device__ __forceinline__ float4 gelu_float4(float4 val) {
    float4 result;
    result.x = gelu_scalar(val.x);
    result.y = gelu_scalar(val.y);
    result.z = gelu_scalar(val.z);
    result.w = gelu_scalar(val.w);
    return result;
}

__global__ void gelu_vectorized_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    
    // Process elements using vectorized loads when possible
    const float4* vec_input = reinterpret_cast<const float4*>(input);
    float4* vec_output = reinterpret_cast<float4*>(output);
    
    int64_t vec_n = n / 4;
    
    // Process 4 elements at a time for better memory bandwidth
    for (int64_t i = idx; i < vec_n; i += stride) {
        float4 in_vec = vec_input[i];
        float4 out_vec = gelu_float4(in_vec);
        vec_output[i] = out_vec;
    }
    
    // Handle remaining elements at the end
    int64_t remainder_start = vec_n * 4;
    for (int64_t i = remainder_start + idx; i < n; i += stride) {
        output[i] = gelu_scalar(input[i]);
    }
}

torch::Tensor gelu_hip(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int64_t n = input.numel();
    
    // Optimized launch configuration for MI300X
    const int threads_per_block = 256;
    
    // Calculate blocks based on vectorized processing
    const int64_t vec_n = n / 4;
    const int64_t num_blocks = min(
        (vec_n + threads_per_block - 1) / threads_per_block,
        static_cast<int64_t>(65535)
    );
    
    // Launch vectorized kernel
    gelu_vectorized_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output;
}
"""

gelu_hip = load_inline(
    name="gelu_hip",
    cpp_sources=gelu_cpp_source,
    functions=["gelu_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu_hip = gelu_hip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu_hip.gelu_hip(x)


def get_inputs():
    batch_size = 4096
    dim = 393216
    # Create tensor on CUDA device
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]


def get_init_inputs():
    return []
