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

// Unroll factor for better instruction-level parallelism
#define UNROLL_FACTOR 4

__device__ __forceinline__ float fast_gelu(float x) {
    // Optimized GELU computation
    float x2 = x * x;
    float x3 = x2 * x;
    float inner = fmaf(COEFF, x3, x) * SQRT_2_OVER_PI;  // fma: x + COEFF * x3
    float tanh_val = tanhf(inner);
    return fmaf(0.5f, x, 0.5f * x * tanh_val);  // 0.5 * x + 0.5 * x * tanh
}

__global__ void gelu_optimized_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    
    // Use vectorized loads for better memory bandwidth
    const float4* vec_input = reinterpret_cast<const float4*>(input);
    float4* vec_output = reinterpret_cast<float4*>(output);
    
    int64_t vec_n = n / 4;
    
    // Grid-stride loop with aggressive unrolling for ILP
    for (int64_t i = idx; i < vec_n; i += stride * UNROLL_FACTOR) {
        #pragma unroll
        for (int j = 0; j < UNROLL_FACTOR; j++) {
            int64_t current_idx = i + j * stride;
            if (current_idx < vec_n) {
                float4 in_vec = vec_input[current_idx];
                float4 out_vec;
                
                out_vec.x = fast_gelu(in_vec.x);
                out_vec.y = fast_gelu(in_vec.y);
                out_vec.z = fast_gelu(in_vec.z);
                out_vec.w = fast_gelu(in_vec.w);
                
                vec_output[current_idx] = out_vec;
            }
        }
    }
    
    // Handle remaining elements
    int64_t remainder_start = vec_n * 4;
    for (int64_t i = remainder_start + idx; i < n; i += stride) {
        output[i] = fast_gelu(input[i]);
    }
}

torch::Tensor gelu_hip(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int64_t n = input.numel();
    
    // Optimized launch configuration for MI300X
    // More blocks for better occupancy
    const int threads_per_block = 256;
    const int64_t vec_n = n / 4;
    
    // Calculate blocks to maximize occupancy while staying within limits
    const int64_t max_blocks = 65535;
    const int64_t target_blocks = min((vec_n + threads_per_block - 1) / threads_per_block, max_blocks);
    
    // Use more blocks than SM count for latency hiding
    const int64_t num_blocks = min(target_blocks * 2, max_blocks);
    
    // Launch optimized kernel
    gelu_optimized_kernel<<<num_blocks, threads_per_block>>>(
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
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]


def get_init_inputs():
    return []
