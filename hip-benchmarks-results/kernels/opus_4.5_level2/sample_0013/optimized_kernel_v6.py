import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Super optimized fused kernel with vectorized memory access
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

// Process 2 batch elements per block for better occupancy
// Each batch element has 512 pooled values, so 1024 threads per block
__global__ void fused_avgpool_gelu_scale_max_2batch_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int out_features,
    float scale_factor,
    float inv_pool_size
) {
    // 1024 threads: first 512 for batch 0, next 512 for batch 1
    int batch_offset = threadIdx.x / 512;
    int pool_idx = threadIdx.x % 512;
    int batch_idx = blockIdx.x * 2 + batch_offset;
    
    float scaled = -INFINITY;
    
    if (batch_idx < batch_size) {
        const float4* row4 = (const float4*)(input + batch_idx * out_features);
        int start4 = pool_idx * 4;
        
        float4 v0 = row4[start4];
        float4 v1 = row4[start4 + 1];
        float4 v2 = row4[start4 + 2];
        float4 v3 = row4[start4 + 3];
        
        float sum = v0.x + v0.y + v0.z + v0.w +
                    v1.x + v1.y + v1.z + v1.w +
                    v2.x + v2.y + v2.z + v2.w +
                    v3.x + v3.y + v3.z + v3.w;
        
        float avg = sum * inv_pool_size;
        float gelu_val = gelu_approx(avg);
        scaled = gelu_val * scale_factor;
    }
    
    __shared__ float sdata[1024];
    sdata[threadIdx.x] = scaled;
    __syncthreads();
    
    // Reduce each half separately
    int base = batch_offset * 512;
    for (int s = 256; s > 0; s >>= 1) {
        if (pool_idx < s) {
            sdata[base + pool_idx] = fmaxf(sdata[base + pool_idx], sdata[base + pool_idx + s]);
        }
        __syncthreads();
    }
    
    if (pool_idx == 0 && batch_idx < batch_size) {
        output[batch_idx] = sdata[base];
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
    
    float inv_pool_size = 1.0f / pool_kernel_size;
    
    // Use 2-batch version for better occupancy
    dim3 grid((batch_size + 1) / 2);
    dim3 block(1024);
    
    fused_avgpool_gelu_scale_max_2batch_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features,
        scale_factor,
        inv_pool_size
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
    name="fused_ops_v6_unique",
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
        x = self.matmul(x)
        x = self.fused_module.fused_avgpool_gelu_scale_max_hip(
            x, self.pool_kernel_size, self.scale_factor
        )
        return x


def get_inputs():
    return [torch.rand(1024, 8192).cuda()]


def get_init_inputs():
    return [8192, 8192, 16, 2.0]
