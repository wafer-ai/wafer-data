import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

gelu_cpp_source = """
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

#define SQRT_2_OVER_PI 0.7978845608028654f
#define COEFF 0.044715f

// Structure for vectorized loads
struct float4_vec {
    float x, y, z, w;
};

__device__ __forceinline__ float gelu_scalar(float x) {
    // Fast GELU computation using tanh approximation
    float x3 = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
    float tanh_val = tanhf(inner);
    return 0.5f * x * (1.0f + tanh_val);
}

__global__ void gelu_vectorized_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    
    // Vectorized version: Process 4 float elements at a time
    const int64_t vectorized_n = n / 4;
    const float4_vec* vec_input = reinterpret_cast<const float4_vec*>(input);
    float4_vec* vec_output = reinterpret_cast<float4_vec*>(output);
    
    if (idx < vectorized_n) {
        float4_vec in_vec = vec_input[idx];
        float4_vec out_vec;
        
        // Process 4 elements in parallel
        out_vec.x = gelu_scalar(in_vec.x);
        out_vec.y = gelu_scalar(in_vec.y);
        out_vec.z = gelu_scalar(in_vec.z);
        out_vec.w = gelu_scalar(in_vec.w);
        
        vec_output[idx] = out_vec;
    }
    
    // Handle remaining elements
    int64_t remainder_start = vectorized_n * 4;
    int64_t remainder_idx = idx + remainder_start;
    if (remainder_idx < n) {
        output[remainder_idx] = gelu_scalar(input[remainder_idx]);
    }
}

torch::Tensor gelu_hip(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int64_t n = input.numel();
    
    // Optimized launch configuration for MI300X with vectorization
    const int threads_per_block = 256;
    const int64_t vectorized_n = n / 4;
    const int64_t num_blocks = min(
        (vectorized_n + threads_per_block - 1) / threads_per_block + 1, 
        static_cast<int64_t>(65535)
    );
    
    // Launch vectorized kernel
    gelu_vectorized_kernel<<<num_blocks, threads_per_block, 0, at::hip::getCurrentHIPStream()>>>(
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
    # Ensure tensor is created on CUDA device with proper alignment for vectorization
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]


def get_init_inputs():
    return []
