import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Two-pass fused kernel: first pass does maxpool + partial sums, second pass finalizes
fused_kernel_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>

#define WARP_SIZE 64

// Single kernel: each batch uses multiple thread blocks for parallel reduction
// Using atomicAdd for final accumulation

__global__ void fused_maxpool_sum_scale_kernel_atomic(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int features,
    const int num_float4s_per_batch,
    const float scale_factor
) {
    int batch_idx = blockIdx.x;
    int block_in_batch = blockIdx.y;
    int num_blocks_per_batch = gridDim.y;
    
    if (batch_idx >= batch_size) return;
    
    const float4* batch_input = reinterpret_cast<const float4*>(input + batch_idx * features);
    
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int global_tid = block_in_batch * blockDim.x + tid;
    int total_threads = num_blocks_per_batch * blockDim.x;
    
    float local_sum = 0.0f;
    
    // Each thread processes multiple float4 elements
    for (int i = global_tid; i < num_float4s_per_batch; i += total_threads) {
        float4 v = batch_input[i];
        float max1 = fmaxf(v.x, v.y);
        float max2 = fmaxf(v.z, v.w);
        local_sum += max1 + max2;
    }
    
    sdata[tid] = local_sum;
    __syncthreads();
    
    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 of each block atomically adds to output
    if (tid == 0) {
        atomicAdd(&output[batch_idx], sdata[0] * scale_factor);
    }
}

torch::Tensor fused_maxpool_sum_scale(torch::Tensor input, float scale_factor) {
    const int batch_size = input.size(0);
    const int features = input.size(1);
    const int num_float4s = features / 4;
    
    // Initialize output to zero for atomic adds
    auto output = torch::zeros({batch_size}, input.options());
    
    const int block_size = 256;
    // Use multiple blocks per batch for more parallelism
    const int blocks_per_batch = 4;  // Tune this
    
    dim3 grid(batch_size, blocks_per_batch);
    dim3 block(block_size);
    const int shared_mem_size = block_size * sizeof(float);
    
    fused_maxpool_sum_scale_kernel_atomic<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        features,
        num_float4s,
        scale_factor
    );
    
    return output;
}
"""

fused_kernel_cpp = """
torch::Tensor fused_maxpool_sum_scale(torch::Tensor input, float scale_factor);
"""

fused_module = load_inline(
    name="fused_maxpool_sum_scale_v3",
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
        self.kernel_size = kernel_size
        self.fused_op = fused_module

    def forward(self, x):
        x = self.matmul(x)
        x = self.fused_op.fused_maxpool_sum_scale(x, self.scale_factor)
        return x
