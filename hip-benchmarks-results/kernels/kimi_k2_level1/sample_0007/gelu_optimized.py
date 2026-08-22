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

__global__ void gelu_kernel(const float* input, float* output, int64_t n) {
    // Grid-stride loop: each thread processes multiple elements
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process 4 elements per thread for better memory bandwidth utilization
    for (int64_t i = idx * 4; i < n; i += blockDim.x * gridDim.x * 4) {
        // Process first element
        if (i < n) {
            float x = input[i];
            float x3 = x * x * x;
            float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
            float tanh_val = tanhf(inner);
            output[i] = 0.5f * x * (1.0f + tanh_val);
        }
        
        // Process second element
        if (i + 1 < n) {
            float x = input[i + 1];
            float x3 = x * x * x;
            float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
            float tanh_val = tanhf(inner);
            output[i + 1] = 0.5f * x * (1.0f + tanh_val);
        }
        
        // Process third element
        if (i + 2 < n) {
            float x = input[i + 2];
            float x3 = x * x * x;
            float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
            float tanh_val = tanhf(inner);
            output[i + 2] = 0.5f * x * (1.0f + tanh_val);
        }
        
        // Process fourth element
        if (i + 3 < n) {
            float x = input[i + 3];
            float x3 = x * x * x;
            float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
            float tanh_val = tanhf(inner);
            output[i + 3] = 0.5f * x * (1.0f + tanh_val);
        }
    }
}

torch::Tensor gelu_hip(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int64_t n = input.numel();
    
    // Configuration for MI300X: 256 threads per block, good occupancy
    const int threads_per_block = 256;
    // Process 4 elements per thread, so we need fewer blocks
    const int64_t total_threads = (n + 3) / 4; // Ceiling division by 4
    const int64_t num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    gelu_kernel<<<num_blocks, threads_per_block>>>(
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
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]


def get_init_inputs():
    return []
