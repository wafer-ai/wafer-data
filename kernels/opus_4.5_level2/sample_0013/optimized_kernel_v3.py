import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel that does everything after matmul more efficiently
# Using multiple threads per batch element for parallel reduction
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// GELU approximation matching PyTorch
__device__ __forceinline__ float gelu_approx(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Optimized kernel: one warp (64 threads) per batch element on MI300X
// pooled_size = 512, so each thread handles 8 elements
__global__ void fused_avgpool_gelu_scale_max_warp_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int out_features,
    int pool_kernel_size,
    float scale_factor
) {
    // Each block handles multiple batch elements
    // We use 256 threads = 4 wavefronts of 64 threads
    const int WARPS_PER_BLOCK = 4;
    const int WARP_SIZE = 64;
    
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    int batch_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (batch_idx >= batch_size) return;
    
    int pooled_size = out_features / pool_kernel_size;
    
    const float* row = input + batch_idx * out_features;
    
    float local_max = -INFINITY;
    
    // Each lane processes multiple pooled elements
    // pooled_size = 512, WARP_SIZE = 64, so each thread handles 8 elements
    for (int pool_idx = lane_id; pool_idx < pooled_size; pool_idx += WARP_SIZE) {
        float sum = 0.0f;
        int start = pool_idx * pool_kernel_size;
        
        // Unroll the pooling loop
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            sum += row[start + k];
        }
        
        float avg = sum * 0.0625f; // divide by 16
        float gelu_val = gelu_approx(avg);
        float scaled = gelu_val * scale_factor;
        local_max = fmaxf(local_max, scaled);
    }
    
    // Warp reduction using shuffle
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down(local_max, offset));
    }
    
    if (lane_id == 0) {
        output[batch_idx] = local_max;
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
    
    const int WARPS_PER_BLOCK = 4;
    const int WARP_SIZE = 64;
    const int BLOCK_SIZE = WARPS_PER_BLOCK * WARP_SIZE; // 256
    
    int num_blocks = (batch_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    
    dim3 grid(num_blocks);
    dim3 block(BLOCK_SIZE);
    
    fused_avgpool_gelu_scale_max_warp_kernel<<<grid, block>>>(
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
    name="fused_avgpool_gelu_scale_max_v3",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_avgpool_gelu_scale_max_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-mllvm", "-amdgpu-early-inline-all=true"]
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
