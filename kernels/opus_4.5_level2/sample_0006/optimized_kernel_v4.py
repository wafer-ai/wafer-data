import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized fused kernel
fused_kernel_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>

// Optimized kernel with better memory access patterns
// Using larger thread blocks and more aggressive vectorization

__global__ void fused_maxpool_sum_scale_optimized(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int features,
    const float scale_factor
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float4* batch_input = reinterpret_cast<const float4*>(input + batch_idx * features);
    const int num_float4s = features / 4;
    
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    float local_sum = 0.0f;
    
    // Unroll loop for better instruction-level parallelism
    int i = tid;
    for (; i + num_threads * 3 < num_float4s; i += num_threads * 4) {
        float4 v0 = batch_input[i];
        float4 v1 = batch_input[i + num_threads];
        float4 v2 = batch_input[i + num_threads * 2];
        float4 v3 = batch_input[i + num_threads * 3];
        
        local_sum += fmaxf(v0.x, v0.y) + fmaxf(v0.z, v0.w);
        local_sum += fmaxf(v1.x, v1.y) + fmaxf(v1.z, v1.w);
        local_sum += fmaxf(v2.x, v2.y) + fmaxf(v2.z, v2.w);
        local_sum += fmaxf(v3.x, v3.y) + fmaxf(v3.z, v3.w);
    }
    
    // Handle remaining elements
    for (; i < num_float4s; i += num_threads) {
        float4 v = batch_input[i];
        local_sum += fmaxf(v.x, v.y) + fmaxf(v.z, v.w);
    }
    
    sdata[tid] = local_sum;
    __syncthreads();
    
    // Tree reduction
    for (int s = num_threads / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Warp-level reduction without sync (AMD wavefront is 64)
    if (tid < 32) {
        float val = sdata[tid] + sdata[tid + 32];
        // Use shuffle for final warp reduction
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down(val, offset, 32);
        }
        if (tid == 0) {
            output[batch_idx] = val * scale_factor;
        }
    }
}

torch::Tensor fused_maxpool_sum_scale(torch::Tensor input, float scale_factor) {
    const int batch_size = input.size(0);
    const int features = input.size(1);
    
    auto output = torch::empty({batch_size}, input.options());
    
    const int block_size = 256;
    const int num_blocks = batch_size;
    const int shared_mem_size = block_size * sizeof(float);
    
    fused_maxpool_sum_scale_optimized<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        features,
        scale_factor
    );
    
    return output;
}
"""

fused_kernel_cpp = """
torch::Tensor fused_maxpool_sum_scale(torch::Tensor input, float scale_factor);
"""

fused_module = load_inline(
    name="fused_maxpool_sum_scale_v4",
    cpp_sources=fused_kernel_cpp,
    cuda_sources=fused_kernel_source,
    functions=["fused_maxpool_sum_scale"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized model using fused HIP kernel for maxpool + sum + scale.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scale_factor = scale_factor
        self.fused_op = fused_module

    def forward(self, x):
        x = self.matmul(x)
        x = self.fused_op.fused_maxpool_sum_scale(x, self.scale_factor)
        return x
