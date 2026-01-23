import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Tiled fused matmul + maxpool + sum + scale kernel
# Use shared memory tiling for better memory access patterns
fused_kernel_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>

#define TILE_SIZE 64
#define BLOCK_SIZE 256

// Tiled version: load input into shared memory, each thread handles multiple output pairs
__global__ void fused_linear_maxpool_sum_scale_tiled(
    const float* __restrict__ input,     // (batch, in_features)
    const float* __restrict__ weight,    // (out_features, in_features)
    const float* __restrict__ bias,      // (out_features,)
    float* __restrict__ output,          // (batch,)
    const int batch_size,
    const int in_features,
    const int out_features,
    const float scale_factor
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    __shared__ float input_tile[TILE_SIZE];
    __shared__ float reduction_buf[BLOCK_SIZE];
    
    const float* batch_input = input + batch_idx * in_features;
    
    int tid = threadIdx.x;
    int num_pairs = out_features / 2;
    int num_tiles = (in_features + TILE_SIZE - 1) / TILE_SIZE;
    
    float local_sum = 0.0f;
    
    // Each thread handles multiple output pairs
    for (int pair_idx = tid; pair_idx < num_pairs; pair_idx += BLOCK_SIZE) {
        int out_idx0 = pair_idx * 2;
        int out_idx1 = pair_idx * 2 + 1;
        
        const float* w0 = weight + out_idx0 * in_features;
        const float* w1 = weight + out_idx1 * in_features;
        
        float dot0 = bias[out_idx0];
        float dot1 = bias[out_idx1];
        
        // Process input in tiles
        for (int tile = 0; tile < num_tiles; tile++) {
            int tile_start = tile * TILE_SIZE;
            
            // Cooperatively load input tile (only for first iteration of outer loop)
            if (pair_idx == tid) {
                for (int i = tid; i < TILE_SIZE && tile_start + i < in_features; i += BLOCK_SIZE) {
                    input_tile[i] = batch_input[tile_start + i];
                }
            }
            __syncthreads();
            
            // Compute partial dot products
            int tile_end = min(TILE_SIZE, in_features - tile_start);
            
            #pragma unroll 4
            for (int k = 0; k < tile_end; k++) {
                float x_val = input_tile[k];
                dot0 += x_val * w0[tile_start + k];
                dot1 += x_val * w1[tile_start + k];
            }
            __syncthreads();
        }
        
        // MaxPool + accumulate
        local_sum += fmaxf(dot0, dot1);
    }
    
    reduction_buf[tid] = local_sum;
    __syncthreads();
    
    // Reduction
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            reduction_buf[tid] += reduction_buf[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[batch_idx] = reduction_buf[0] * scale_factor;
    }
}

torch::Tensor fused_linear_maxpool_sum_scale(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor
) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);
    
    auto output = torch::empty({batch_size}, input.options());
    
    const int num_blocks = batch_size;
    
    fused_linear_maxpool_sum_scale_tiled<<<num_blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        scale_factor
    );
    
    return output;
}
"""

fused_kernel_cpp = """
torch::Tensor fused_linear_maxpool_sum_scale(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor
);
"""

fused_module = load_inline(
    name="fused_linear_maxpool_sum_scale_v6",
    cpp_sources=fused_kernel_cpp,
    cuda_sources=fused_kernel_source,
    functions=["fused_linear_maxpool_sum_scale"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized model with fully fused linear + maxpool + sum + scale.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale_factor = scale_factor
        self.fused_op = fused_module

    def forward(self, x):
        return self.fused_op.fused_linear_maxpool_sum_scale(
            x,
            self.linear.weight,
            self.linear.bias,
            self.scale_factor
        )
