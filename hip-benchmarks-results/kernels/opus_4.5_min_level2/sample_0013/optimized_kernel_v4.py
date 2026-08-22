import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Let's try using the linear output directly without intermediate storage
# and implementing a fused kernel with async memory handling
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

// Super optimized fused kernel using vectorized loads
// Process multiple batches per block for better GPU utilization
__global__ void fused_avgpool_gelu_scale_max_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int out_features,
    const float scale_factor,
    const float inv_pool_size
) {
    // pool_kernel_size is 16, so pooled_size = out_features / 16
    constexpr int POOL_SIZE = 16;
    const int pooled_size = out_features >> 4;  // / 16
    
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    const float* batch_input = input + batch_idx * out_features;
    
    // Use float4 vectorized loads
    const float4* batch_input_vec = reinterpret_cast<const float4*>(batch_input);
    
    float local_max = -FLT_MAX;
    
    // Each thread processes multiple pooled elements
    // pooled_size = 512 for out_features = 8192
    for (int pool_idx = tid; pool_idx < pooled_size; pool_idx += num_threads) {
        float sum = 0.0f;
        
        // Each pooled element spans 16 floats = 4 float4
        int vec_start = pool_idx << 2;  // * 4
        
        float4 v0 = batch_input_vec[vec_start];
        float4 v1 = batch_input_vec[vec_start + 1];
        float4 v2 = batch_input_vec[vec_start + 2];
        float4 v3 = batch_input_vec[vec_start + 3];
        
        sum = v0.x + v0.y + v0.z + v0.w +
              v1.x + v1.y + v1.z + v1.w +
              v2.x + v2.y + v2.z + v2.w +
              v3.x + v3.y + v3.z + v3.w;
        
        float avg = sum * inv_pool_size;
        float result = gelu(avg) * scale_factor;
        local_max = fmaxf(local_max, result);
    }
    
    // Block-level reduction using shared memory
    extern __shared__ float sdata[];
    sdata[tid] = local_max;
    __syncthreads();
    
    // Parallel reduction
    for (int s = num_threads >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
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
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(pool_kernel_size == 16, "Only pool_kernel_size=16 is supported");
    
    const int batch_size = input.size(0);
    const int out_features = input.size(1);
    
    auto output = torch::empty({batch_size}, input.options());
    
    // Pooled size = 512, so 256 threads is efficient
    const int block_size = 256;
    const int num_blocks = batch_size;
    const float inv_pool_size = 1.0f / pool_kernel_size;
    const int shared_mem_size = block_size * sizeof(float);
    
    fused_avgpool_gelu_scale_max_kernel<<<num_blocks, block_size, shared_mem_size>>>(
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

fused_kernel_cpp = """
torch::Tensor fused_avgpool_gelu_scale_max_hip(
    torch::Tensor input,
    int pool_kernel_size,
    float scale_factor
);
"""

fused_module = load_inline(
    name="fused_avgpool_gelu_scale_max_v4",
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
            x.contiguous(), self.pool_kernel_size, self.scale_factor
        )
        return x


def get_inputs():
    return [torch.rand(1024, 8192).cuda()]


def get_init_inputs():
    return [8192, 8192, 16, 2.0]
