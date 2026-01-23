import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel for AvgPool + GELU + Scale + Max
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ __forceinline__ float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Fused kernel: AvgPool1d + GELU + Scale + Max reduction
// Input: (batch_size, out_features)
// Output: (batch_size,)
__global__ void fused_avgpool_gelu_scale_max_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int out_features,
    int pool_kernel_size,
    float scale_factor
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    int pooled_size = out_features / pool_kernel_size;
    
    // Each thread processes multiple pooled elements
    float local_max = -INFINITY;
    
    const float* row = input + batch_idx * out_features;
    
    for (int pool_idx = tid; pool_idx < pooled_size; pool_idx += blockDim.x) {
        // Compute average pooling
        float sum = 0.0f;
        int start = pool_idx * pool_kernel_size;
        
        #pragma unroll 4
        for (int k = 0; k < pool_kernel_size; k++) {
            sum += row[start + k];
        }
        float avg = sum / (float)pool_kernel_size;
        
        // Apply GELU
        float gelu_val = gelu(avg);
        
        // Apply scale
        float scaled = gelu_val * scale_factor;
        
        // Track local max
        local_max = fmaxf(local_max, scaled);
    }
    
    // Warp reduction for max
    __shared__ float shared_max[256];
    shared_max[tid] = local_max;
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[batch_idx] = shared_max[0];
    }
}

torch::Tensor fused_avgpool_gelu_scale_max_hip(
    torch::Tensor input,
    int pool_kernel_size,
    float scale_factor
) {
    int batch_size = input.size(0);
    int out_features = input.size(1);
    
    auto output = torch::empty({batch_size}, input.options());
    
    const int block_size = 256;
    dim3 grid(batch_size);
    dim3 block(block_size);
    
    fused_avgpool_gelu_scale_max_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features,
        pool_kernel_size,
        scale_factor
    );
    
    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_avgpool_gelu_scale_max_hip(
    torch::Tensor input,
    int pool_kernel_size,
    float scale_factor
);
"""

fused_module = load_inline(
    name="fused_avgpool_gelu_scale_max",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_avgpool_gelu_scale_max_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    """
    Optimized model with fused AvgPool+GELU+Scale+Max kernel.
    """
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor
        self.fused_module = fused_module

    def forward(self, x):
        # Use PyTorch's optimized linear layer
        x = self.matmul(x)
        # Fused kernel for the rest
        x = self.fused_module.fused_avgpool_gelu_scale_max_hip(
            x, self.pool_kernel_size, self.scale_factor
        )
        return x


def get_inputs():
    return [torch.rand(1024, 8192).cuda()]


def get_init_inputs():
    return [8192, 8192, 16, 2.0]
