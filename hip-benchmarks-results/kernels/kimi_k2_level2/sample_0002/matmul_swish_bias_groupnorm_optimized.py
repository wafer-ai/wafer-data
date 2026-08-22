import torch
import torch.nn as nn
import os
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized but simpler kernel for better correctness
matmul_swish_bias_source = """
#include <hip/hip_runtime.h>
#include <math.h>

#define TILE_SIZE_M 64
#define TILE_SIZE_N 64
#define TILE_SIZE_K 16

__global__ void fused_matmul_swish_bias_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ linear_bias,
    const float* __restrict__ extra_bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features) {
    
    __shared__ float s_x[TILE_SIZE_M][TILE_SIZE_K];
    __shared__ float s_W[TILE_SIZE_K][TILE_SIZE_N];
    
    int row = blockIdx.y * TILE_SIZE_M + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE_N + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    float sum = 0.0f;
    
    // Loop over K dimension (in_features)
    for (int k = 0; k < in_features; k += TILE_SIZE_K) {
        // Load tile of x: x[row, k + tx]
        if (row < batch_size && k + tx < in_features) {
            s_x[ty][tx] = x[row * in_features + (k + tx)];
        } else {
            s_x[ty][tx] = 0.0f;
        }
        
        // Load tile of W: W[col, k + ty]
        // weight shape: (out_features, in_features)
        if (col < out_features && k + ty < in_features) {
            s_W[ty][tx] = weight[col * in_features + (k + ty)];
        } else {
            s_W[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        #pragma unroll 8
        for (int i = 0; i < TILE_SIZE_K; i++) {
            sum += s_x[ty][i] * s_W[i][tx];
        }
        
        __syncthreads();
    }
    
    if (row < batch_size && col < out_features) {
        // Add linear bias
        sum += linear_bias[col];
        
        // Swish activation: x * sigmoid(x)
        float sigmoid = 1.0f / (1.0f + expf(-sum));
        float swish = sum * sigmoid;
        
        // Add extra bias
        output[row * out_features + col] = swish + extra_bias[col];
    }
}

torch::Tensor fused_matmul_swish_bias_hip(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor linear_bias,
    torch::Tensor extra_bias) {
    
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, x.options());
    
    dim3 block(TILE_SIZE_N, TILE_SIZE_M);
    dim3 grid((out_features + TILE_SIZE_N - 1) / TILE_SIZE_N, 
              (batch_size + TILE_SIZE_M - 1) / TILE_SIZE_M);
    
    fused_matmul_swish_bias_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        linear_bias.data_ptr<float>(),
        extra_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features);
    
    return output;
}
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=matmul_swish_bias_source,
    functions=["fused_matmul_swish_bias_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        
        # Use Linear module for proper weight initialization
        self.linear = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        
    def forward(self, x):
        # Use optimized fused kernel for matmul + Swish + bias
        x = fused_ops.fused_matmul_swish_bias_hip(x, self.linear.weight, self.linear.bias, self.bias)
        
        # Apply GroupNorm
        x = self.group_norm(x)
        return x

def get_inputs():
    return [torch.rand(32768, 1024)]

def get_init_inputs():
    return [1024, 4096, 64, (4096,)]