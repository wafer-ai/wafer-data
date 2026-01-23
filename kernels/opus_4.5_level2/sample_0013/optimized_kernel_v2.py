import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel for AvgPool + GELU + Scale + Max with vectorized loads
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

// Warp-level reduction for max
__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 32; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down(val, offset));
    }
    return val;
}

// Block-level reduction for max
__device__ __forceinline__ float blockReduceMax(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 64;
    int wid = threadIdx.x / 64;
    
    // Warp-level reduction
    for (int offset = 32; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down(val, offset));
    }
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    // Final reduction in first warp
    val = (threadIdx.x < blockDim.x / 64) ? shared[lane] : -INFINITY;
    if (wid == 0) {
        for (int offset = 32; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down(val, offset));
        }
    }
    return val;
}

// Optimized fused kernel with vectorized loads (float4)
// pool_kernel_size = 16 allows us to read 4 float4s per pooled element
__global__ void fused_avgpool_gelu_scale_max_kernel_v2(
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
    
    // Using vectorized loads when pool_kernel_size is 16 (divisible by 4)
    const float4* row4 = (const float4*)row;
    
    for (int pool_idx = tid; pool_idx < pooled_size; pool_idx += blockDim.x) {
        float sum = 0.0f;
        
        // Each pooled element spans pool_kernel_size/4 = 4 float4s
        int start4 = pool_idx * (pool_kernel_size / 4);
        
        #pragma unroll
        for (int k = 0; k < pool_kernel_size / 4; k++) {
            float4 v = row4[start4 + k];
            sum += v.x + v.y + v.z + v.w;
        }
        
        float avg = sum / (float)pool_kernel_size;
        float gelu_val = gelu(avg);
        float scaled = gelu_val * scale_factor;
        local_max = fmaxf(local_max, scaled);
    }
    
    // Shared memory reduction
    __shared__ float shared_max[512];
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
    
    // Use 512 threads for better occupancy
    const int block_size = 512;
    dim3 grid(batch_size);
    dim3 block(block_size);
    
    fused_avgpool_gelu_scale_max_kernel_v2<<<grid, block>>>(
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
    name="fused_avgpool_gelu_scale_max_v2",
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
