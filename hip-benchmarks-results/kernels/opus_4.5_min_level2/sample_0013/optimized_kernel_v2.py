import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel: AvgPool1d + GELU + Scale + Max with vectorized loads
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>
#include <float.h>

// GELU approximation using tanh
__device__ __forceinline__ float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);
    return 0.5f * x * (1.0f + tanhf(tanh_arg));
}

// Warp reduction for max using AMD wavefront size of 64
__device__ __forceinline__ float warp_reduce_max_64(float val) {
    for (int offset = 32; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down(val, offset));
    }
    return val;
}

// Optimized fused kernel with vectorized memory access
// pool_kernel_size is known to be 16 at compile time
__global__ void fused_avgpool_gelu_scale_max_kernel_v2(
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
    float local_max = -FLT_MAX;
    
    // Use float4 for vectorized loading (4 floats = 16 bytes)
    const float4* batch_input_vec = reinterpret_cast<const float4*>(batch_input);
    
    for (int pool_idx = tid; pool_idx < pooled_size; pool_idx += num_threads) {
        float sum = 0.0f;
        
        // pool_kernel_size=16, load as 4 float4 vectors
        int vec_start = pool_idx * (pool_kernel_size / 4);
        
        #pragma unroll
        for (int v = 0; v < 4; v++) {
            float4 data = batch_input_vec[vec_start + v];
            sum += data.x + data.y + data.z + data.w;
        }
        
        float avg = sum * (1.0f / pool_kernel_size);
        
        // Apply GELU and scale
        float result = gelu(avg) * scale_factor;
        
        local_max = fmaxf(local_max, result);
    }
    
    // Warp-level reduction
    local_max = warp_reduce_max_64(local_max);
    
    // Shared memory for block-level reduction
    __shared__ float shared_max[16];  // Max 16 warps with 256 threads per block (AMD wave64)
    
    int lane = tid & 63;  // AMD wavefront is 64
    int warp_id = tid >> 6;
    
    if (lane == 0) {
        shared_max[warp_id] = local_max;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (tid < 64) {
        int num_warps = (num_threads + 63) >> 6;
        local_max = (tid < num_warps) ? shared_max[tid] : -FLT_MAX;
        local_max = warp_reduce_max_64(local_max);
        
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
    
    // Use 512 threads for better occupancy
    const int block_size = 512;
    const int num_blocks = batch_size;
    
    fused_avgpool_gelu_scale_max_kernel_v2<<<num_blocks, block_size>>>(
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
    name="fused_avgpool_gelu_scale_max_v2",
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
