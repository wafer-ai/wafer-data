import os
import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

matmul_div_gelu_cpp_source = """
#include <hip/hip_runtime.h>

#define BLOCK_SIZE_M 16
#define BLOCK_SIZE_N 16
#define BLOCK_SIZE_K 64

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
}

__global__ void matmul_div_gelu_kernel_tiled(
    const float* input,      // [batch_size, input_size]
    const float* weight,     // [output_size, input_size]
    const float* bias,       // [output_size]
    float* output,           // [batch_size, output_size]
    int batch_size,
    int input_size,
    int output_size,
    float divisor
) {
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    
    int row = by * BLOCK_SIZE_M + ty;
    int col = bx * BLOCK_SIZE_N + tx;
    
    __shared__ float tile_input[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float tile_weight[BLOCK_SIZE_K][BLOCK_SIZE_N];
    
    float sum = 0.0f;
    
    for (int k = 0; k < (input_size + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K; k++) {
        // Load tile of input
        int input_col = k * BLOCK_SIZE_K + tx;
        if (input_col < input_size && row < batch_size) {
            tile_input[ty][tx] = input[row * input_size + input_col];
        } else {
            tile_input[ty][tx] = 0.0f;
        }
        
        // Load tile of weight (transposed)
        int weight_row = k * BLOCK_SIZE_K + ty;
        int weight_col = bx * BLOCK_SIZE_N + tx;
        if (weight_row < input_size && weight_col < output_size) {
            tile_weight[ty][tx] = weight[weight_col * input_size + weight_row];
        } else {
            tile_weight[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Unroll computation 
        #pragma unroll
        for (int kk = 0; kk < BLOCK_SIZE_K; kk++) {
            sum += tile_input[ty][kk] * tile_weight[kk][tx];
        }
        
        __syncthreads();
    }
    
    // Write result with fused operations
    if (row < batch_size && col < output_size) {
        sum += bias[col];
        sum = gelu(sum / divisor);
        output[row * output_size + col] = sum;
    }
}

torch::Tensor matmul_div_gelu_hip(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor
) {
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int output_size = weight.size(0);
    
    auto output = torch::zeros({batch_size, output_size}, input.options());
    
    dim3 blockDim(BLOCK_SIZE_N, BLOCK_SIZE_M);
    dim3 gridDim(
        (output_size + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N,
        (batch_size + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M
    );
    
    matmul_div_gelu_kernel_tiled<<<gridDim, blockDim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        output_size,
        divisor
    );
    
    return output;
}
"""

matmul_div_gelu = load_inline(
    name="matmul_div_gelu",
    cpp_sources=matmul_div_gelu_cpp_source,
    functions=["matmul_div_gelu_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model with fused matmul + divide + GELU kernel using tiling
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.divisor = divisor
        self.weight = nn.Parameter(torch.Tensor(output_size, input_size))
        self.bias = nn.Parameter(torch.Tensor(output_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        self.matmul_div_gelu = matmul_div_gelu

    def forward(self, x):
        return self.matmul_div_gelu.matmul_div_gelu_hip(
            x, self.weight, self.bias, self.divisor
        )