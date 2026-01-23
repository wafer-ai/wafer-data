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

__global__ void gelu_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t n) {
    // Use int64_t for indexing to handle large tensors
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    
    // Grid-stride loop for better memory bandwidth utilization
    for (int64_t i = idx; i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
        float tanh_val = tanhf(inner);
        output[i] = 0.5f * x * (1.0f + tanh_val);
    }
}

torch::Tensor gelu_hip(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int64_t n = input.numel();
    
    // Optimized launch configuration for MI300X
    const int threads_per_block = 256;
    // Calculate number of blocks to cover the entire tensor
    const int64_t num_blocks = min((n + threads_per_block - 1) / threads_per_block, static_cast<int64_t>(65535));
    
    // Launch kernel with grid-stride loop
    gelu_kernel<<<num_blocks, threads_per_block, 0, at::hip::getCurrentHIPStream()>>>(
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
    # Ensure tensor is created on CUDA device
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]


def get_init_inputs():
    return []
