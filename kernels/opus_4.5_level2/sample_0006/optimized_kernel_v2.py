import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused kernel for MaxPool1d + Sum + Scale using vectorized loads
fused_kernel_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>

// Fused MaxPool1d (kernel_size=2) + Sum + Scale kernel with vectorized loads
// Each block handles one batch element using float4 for coalesced memory access

__global__ void fused_maxpool_sum_scale_kernel_v2(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int features,
    const float scale_factor
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float4* batch_input = reinterpret_cast<const float4*>(input + batch_idx * features);
    const int num_float4s = features / 4;  // Number of float4 elements
    
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    float local_sum = 0.0f;
    
    // Each float4 contains 4 floats
    // MaxPool with kernel_size=2 means: max(v.x, v.y) + max(v.z, v.w) for each float4
    for (int i = tid; i < num_float4s; i += num_threads) {
        float4 v = batch_input[i];
        float max1 = fmaxf(v.x, v.y);
        float max2 = fmaxf(v.z, v.w);
        local_sum += max1 + max2;
    }
    
    sdata[tid] = local_sum;
    __syncthreads();
    
    // Parallel reduction
    for (int s = num_threads / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Warp-level reduction (no sync needed within a warp)
    if (tid < 32) {
        volatile float* vsdata = sdata;
        if (num_threads >= 64) vsdata[tid] += vsdata[tid + 32];
        if (num_threads >= 32) vsdata[tid] += vsdata[tid + 16];
        if (num_threads >= 16) vsdata[tid] += vsdata[tid + 8];
        if (num_threads >= 8) vsdata[tid] += vsdata[tid + 4];
        if (num_threads >= 4) vsdata[tid] += vsdata[tid + 2];
        if (num_threads >= 2) vsdata[tid] += vsdata[tid + 1];
    }
    
    if (tid == 0) {
        output[batch_idx] = sdata[0] * scale_factor;
    }
}

torch::Tensor fused_maxpool_sum_scale(torch::Tensor input, float scale_factor) {
    const int batch_size = input.size(0);
    const int features = input.size(1);
    
    auto output = torch::empty({batch_size}, input.options());
    
    const int block_size = 256;
    const int num_blocks = batch_size;
    const int shared_mem_size = block_size * sizeof(float);
    
    fused_maxpool_sum_scale_kernel_v2<<<num_blocks, block_size, shared_mem_size>>>(
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
    name="fused_maxpool_sum_scale_v2",
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
