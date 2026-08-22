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

// Large vectorized GELU kernel - each thread processes 8 floats
__global__ void gelu_kernel_optimized(const float* __restrict__ input, 
                                       float* __restrict__ output,
                                       int total_size) {
    // Each thread handles 8 elements for better memory throughput
    const int elements_per_thread = 8;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * elements_per_thread;
    
    // Process full 8-element chunks
    if (idx + elements_per_thread <= total_size) {
        // Load 8 floats (2 float4s)
        float4 in1 = *reinterpret_cast<const float4*>(input + idx);
        float4 in2 = *reinterpret_cast<const float4*>(input + idx + 4);
        
        // Compute GELU
        float4 out1, out2;
        out1.x = gelu_exact(in1.x);
        out1.y = gelu_exact(in1.y);
        out1.z = gelu_exact(in1.z);
        out1.w = gelu_exact(in1.w);
        out2.x = gelu_exact(in2.x);
        out2.y = gelu_exact(in2.y);
        out2.z = gelu_exact(in2.z);
        out2.w = gelu_exact(in2.w);
        
        // Store 8 floats
        *reinterpret_cast<float4*>(output + idx) = out1;
        *reinterpret_cast<float4*>(output + idx + 4) = out2;
    }
    // Handle boundary cases
    else if (idx < total_size) {
        for (int i = idx; i < total_size && i < idx + elements_per_thread; i++) {
            output[i] = gelu_exact(input[i]);
        }
    }
}

torch::Tensor gelu_hip(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
    
    auto output = torch::empty_like(input);
    int total_size = input.numel();
    
    const int elements_per_thread = 8;
    const int block_size = 256;
    
    // Calculate number of threads needed
    int num_threads_needed = (total_size + elements_per_thread - 1) / elements_per_thread;
    int num_blocks = (num_threads_needed + block_size - 1) / block_size;
    
    gelu_kernel_optimized<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        total_size
    );
    
    return output;
}
"""

gelu_cpp_source = """
torch::Tensor gelu_hip(torch::Tensor input);
"""

gelu_module = load_inline(
    name="gelu_hip_v3",
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
