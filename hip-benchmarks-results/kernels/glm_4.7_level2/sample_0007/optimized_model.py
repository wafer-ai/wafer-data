import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

matmul_swish_scale_cpp_source = """
#include <hip/hip_runtime.h>

// Tile sizes for shared memory tiling
// Use square tiles for simplicity
#define TILE_M 8
#define TILE_N 8
#define TILE_K 32

// Fused kernel: matmul + swish activation + scaling with tiling
__global__ void matmul_swish_scale_tiled_kernel(
    const float* input,      // [batch_size, in_features]
    const float* weight,     // [out_features, in_features]
    const float* bias,       // [out_features] (can be nullptr)
    float* output,           // [batch_size, out_features]
    int batch_size,
    int in_features,
    int out_features,
    float scaling_factor
) {
    // Block computes TILE_M x TILE_N tile of output
    int batch_start = blockIdx.y * TILE_M;
    int out_start = blockIdx.x * TILE_N;
    
    // Each thread computes one output element
    int batch_idx = batch_start + threadIdx.y;
    int out_idx = out_start + threadIdx.x;
    
    // Shared memory tiles
    __shared__ float input_tile[TILE_M][TILE_K];
    __shared__ float weight_tile[TILE_N][TILE_K];
    
    // Accumulator
    float sum = 0.0f;
    
    // Loop over K dimension in tiles
    for (int k_tile = 0; k_tile < in_features; k_tile += TILE_K) {
        // Load input tile: input_tile[m][k]
        // Each thread (x,y) loads one element
        int k_idx = k_tile + threadIdx.y;  // Use threadIdx.y for K dimension in loading
        int load_batch_idx = batch_start + threadIdx.x;  // Use threadIdx.x for M dimension
        
        // Need to carefully map: blockDim.x = TILE_N=8, blockDim.y = TILE_M=8
        // For input tile (TILE_M x TILE_K = 8x32):
        // We need 8*32 = 256 loads, but block has 64 threads
        // Solution: load multiple elements per thread or use iterative loading
        
        // Simplified: Only load elements that fit in our block and pad with zeros
        // Since TILE_K=32 > TILE_M=8, we need to handle this
        // Let's loop to load all elements
        
        // Strategy: threadIdx.x selects which row of input_tile to load (0-7), threadIdx.y selects initial offset in K
        // Then load 4 elements per thread to cover TILE_K=32
        for (int k_offset = 0; k_offset < TILE_K; k_offset += (TILE_M * TILE_N) / TILE_N) {
            int actual_k = k_tile + threadIdx.y + k_offset;
            int actual_batch = batch_start + threadIdx.x;
            
            if (threadIdx.x < TILE_M && actual_k < TILE_K) {
                if (actual_batch < batch_size && actual_k < in_features) {
                    input_tile[threadIdx.x][actual_k] = input[actual_batch * in_features + actual_k];
                } else {
                    input_tile[threadIdx.x][actual_k] = 0.0f;
                }
            }
        }
        
        // Load weight tile: weight_tile[n][k] where n=0..7, k=0..31
        // weight is [out_features, in_features]
        for (int k_offset = 0; k_offset < TILE_K; k_offset += (TILE_M * TILE_N) / TILE_M) {
            int actual_k = k_tile + threadIdx.x + k_offset;  // Use threadIdx.x for K
            int actual_out = out_start + threadIdx.y;  // Use threadIdx.y for N dimension
            
            if (threadIdx.y < TILE_N && actual_k < TILE_K) {
                if (actual_out < out_features && actual_k < in_features) {
                    weight_tile[threadIdx.y][actual_k] = weight[actual_out * in_features + actual_k];
                } else {
                    weight_tile[threadIdx.y][actual_k] = 0.0f;
                }
            }
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        // output[b][n] = sum_k input_tile[b][k] * weight_tile[n][k]
        if (threadIdx.x < TILE_N && threadIdx.y < TILE_M) {
            #pragma unroll
            for (int k = 0; k < TILE_K; k++) {
                // input_tile[threadIdx.y][k] * weight_tile[threadIdx.x][k]
                sum += input_tile[threadIdx.y][k] * weight_tile[threadIdx.x][k];
            }
        }
        
        __syncthreads();
    }
    
    // Apply activation and scaling if in bounds
    if (batch_idx < batch_size && out_idx < out_features) {
        // Add bias if available
        if (bias != nullptr) {
            sum += bias[out_idx];
        }
        
        // Apply Swish activation: x * sigmoid(x)
        float sigmoid_val = 1.0f / (1.0f + __expf(-sum));
        float swish_val = sum * sigmoid_val;
        
        // Apply scaling
        float result = swish_val * scaling_factor;
        
        // Write output
        output[batch_idx * out_features + out_idx] = result;
    }
}

torch::Tensor matmul_swish_scale_hip(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor
) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);  // weight is [out_features, in_features]
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    // Launch kernel
    dim3 block_dim(TILE_N, TILE_M);  // blockDim.x = TILE_N, blockDim.y = TILE_M
    dim3 grid_dim((out_features + TILE_N - 1) / TILE_N,
                  (batch_size + TILE_M - 1) / TILE_M);
    
    hipLaunchKernelGGL(
        matmul_swish_scale_tiled_kernel,
        grid_dim, block_dim, 0, 0,
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        scaling_factor
    );
    
    return output;
}
"""

matmul_swish_scale = load_inline(
    name="matmul_swish_scale",
    cpp_sources=matmul_swish_scale_cpp_source,
    functions=["matmul_swish_scale_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model with fused matmul + Swish + scaling kernel using tiled GEMM
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.matmul_swish_scale = matmul_swish_scale

    def forward(self, x):
        # Use fused kernel for matmul + Swish + scale
        x = self.matmul_swish_scale.matmul_swish_scale_hip(
            x,
            self.matmul.weight,
            self.matmul.bias,
            self.scaling_factor
        )
        return x