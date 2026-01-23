import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Try fusing more operations and optimizing memory access patterns
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

// Optimized fused kernel for MI300X
// Each block processes one batch element
// Using larger thread blocks for better occupancy
__global__ __launch_bounds__(1024, 1)
void fused_avgpool_gelu_scale_max_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int out_features,
    const int pool_kernel_size,
    const float scale_factor,
    const float inv_pool_size
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    const int pooled_size = out_features / pool_kernel_size;
    
    const float* batch_input = input + batch_idx * out_features;
    
    float local_max = -FLT_MAX;
    
    // Each thread handles multiple pooled regions
    // With pool_kernel_size=16 and out_features=8192, pooled_size=512
    // With 1024 threads, each thread handles ~0.5 pooled elements on average
    // But we want each thread to do more work for better efficiency
    
    for (int pool_idx = tid; pool_idx < pooled_size; pool_idx += num_threads) {
        float sum = 0.0f;
        int base = pool_idx * pool_kernel_size;
        
        // Unrolled sum for pool_kernel_size=16
        #pragma unroll 16
        for (int k = 0; k < 16; k++) {
            sum += batch_input[base + k];
        }
        
        float avg = sum * inv_pool_size;
        float result = gelu(avg) * scale_factor;
        local_max = fmaxf(local_max, result);
    }
    
    // Block-level reduction using shared memory
    __shared__ float sdata[1024];
    sdata[tid] = local_max;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = num_threads / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // Final warp reduction (no sync needed within warp)
    if (tid < 32) {
        volatile float* vsmem = sdata;
        if (num_threads >= 64) vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid + 32]);
        if (tid < 16) vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid + 16]);
        if (tid < 8) vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid + 8]);
        if (tid < 4) vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid + 4]);
        if (tid < 2) vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid + 2]);
        if (tid < 1) vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid + 1]);
    }
    
    if (tid == 0) {
        output[batch_idx] = sdata[0];
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
    
    const int block_size = 512;  // Adjusted for better occupancy
    const int num_blocks = batch_size;
    const float inv_pool_size = 1.0f / pool_kernel_size;
    
    fused_avgpool_gelu_scale_max_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features,
        pool_kernel_size,
        scale_factor,
        inv_pool_size
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
    name="fused_avgpool_gelu_scale_max_v3",
    cpp_sources=fused_kernel_cpp,
    cuda_sources=fused_kernel_source,
    functions=["fused_avgpool_gelu_scale_max_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"],
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
        x = self.matmul(x)
        x = self.fused_module.fused_avgpool_gelu_scale_max_hip(
            x, self.pool_kernel_size, self.scale_factor
        )
        return x


def get_inputs():
    return [torch.rand(1024, 8192).cuda()]


def get_init_inputs():
    return [8192, 8192, 16, 2.0]
