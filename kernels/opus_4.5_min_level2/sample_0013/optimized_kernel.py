import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel: AvgPool1d + GELU + Scale + Max
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// GELU approximation using tanh
__device__ __forceinline__ float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);
    return 0.5f * x * (1.0f + tanhf(tanh_arg));
}

// Warp reduction for max
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 32; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down(val, offset));
    }
    return val;
}

// Fused AvgPool + GELU + Scale + Max kernel
// Input: (batch_size, out_features)
// Output: (batch_size,)
__global__ void fused_avgpool_gelu_scale_max_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int out_features,
    const int pool_kernel_size,
    const float scale_factor
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    // After pooling, we have out_features / pool_kernel_size elements
    const int pooled_size = out_features / pool_kernel_size;
    
    const float* batch_input = input + batch_idx * out_features;
    
    // Each thread processes multiple pooled elements
    float local_max = -INFINITY;
    
    for (int pool_idx = tid; pool_idx < pooled_size; pool_idx += num_threads) {
        // Compute average over pool_kernel_size elements
        float sum = 0.0f;
        int start_idx = pool_idx * pool_kernel_size;
        
        #pragma unroll 4
        for (int k = 0; k < pool_kernel_size; k++) {
            sum += batch_input[start_idx + k];
        }
        float avg = sum / (float)pool_kernel_size;
        
        // Apply GELU
        float gelu_val = gelu(avg);
        
        // Apply scale
        float scaled_val = gelu_val * scale_factor;
        
        // Update local max
        local_max = fmaxf(local_max, scaled_val);
    }
    
    // Warp-level reduction
    local_max = warp_reduce_max(local_max);
    
    // Shared memory for block-level reduction
    __shared__ float shared_max[32];
    
    int lane = tid % 64;
    int warp_id = tid / 64;
    
    if (lane == 0) {
        shared_max[warp_id] = local_max;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (tid < 32) {
        local_max = (tid < (num_threads + 63) / 64) ? shared_max[tid] : -INFINITY;
        local_max = warp_reduce_max(local_max);
        
        if (tid == 0) {
            output[batch_idx] = local_max;
        }
    }
}

torch::Tensor fused_avgpool_gelu_scale_max_hip(
    torch::Tensor input,
    int pool_kernel_size,
    float scale_factor
) {
    const int batch_size = input.size(0);
    const int out_features = input.size(1);
    
    auto output = torch::empty({batch_size}, input.options());
    
    const int block_size = 256;
    const int num_blocks = batch_size;
    
    fused_avgpool_gelu_scale_max_kernel<<<num_blocks, block_size>>>(
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

fused_kernel_cpp = """
torch::Tensor fused_avgpool_gelu_scale_max_hip(
    torch::Tensor input,
    int pool_kernel_size,
    float scale_factor
);
"""

fused_module = load_inline(
    name="fused_avgpool_gelu_scale_max",
    cpp_sources=fused_kernel_cpp,
    cuda_sources=fused_kernel_source,
    functions=["fused_avgpool_gelu_scale_max_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized model implementing "Matmul_AvgPool_GELU_Scale_Max" with fused kernel.
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
        # Fused AvgPool + GELU + Scale + Max
        x = self.fused_module.fused_avgpool_gelu_scale_max_hip(
            x, self.pool_kernel_size, self.scale_factor
        )
        return x


def get_inputs():
    return [torch.rand(1024, 8192).cuda()]


def get_init_inputs():
    return [8192, 8192, 16, 2.0]
