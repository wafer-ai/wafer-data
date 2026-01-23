import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

cpp_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>

#define BLOCK_SIZE 32
#define TILE_SIZE 32

__global__ void linear_div_gelu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int input_size,
    int output_size,
    float divisor,
    float sqrt_2_over_pi,
    float c
) {
    __shared__ float input_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float weight_tile[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < output_size) {
        float sum = 0.0f;
        
        // Matrix multiplication: input @ weight^T
        for (int tile_idx = 0; tile_idx < (input_size + TILE_SIZE - 1) / TILE_SIZE; ++tile_idx) {
            // Load input tile
            int input_col = tile_idx * TILE_SIZE + threadIdx.x;
            if (input_col < input_size) {
                input_tile[threadIdx.y][threadIdx.x] = input[row * input_size + input_col];
            } else {
                input_tile[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            // Load weight tile (weight is in shape (output_size, input_size))
            int weight_row = tile_idx * TILE_SIZE + threadIdx.y;
            if (weight_row < input_size) {
                weight_tile[threadIdx.y][threadIdx.x] = weight[col * input_size + weight_row];
            } else {
                weight_tile[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            __syncthreads();
            
            // Compute partial sum
            for (int k = 0; k < TILE_SIZE; ++k) {
                sum += input_tile[threadIdx.y][k] * weight_tile[k][threadIdx.x];
            }
            
            __syncthreads();
        }
        
        // Add bias
        sum += bias[col];
        
        // Divide by divisor
        sum /= divisor;
        
        // Apply GELU activation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        float x_cubed = sum * sum * sum;
        float tanh_arg = sqrt_2_over_pi * (sum + c * x_cubed);
        float tanh_result = tanhf(tanh_arg);
        float gelu_result = sum * 0.5f * (1.0f + tanh_result);
        
        output[row * output_size + col] = gelu_result;
    }
}

torch::Tensor linear_div_gelu(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float divisor) {
    // input shape: (batch_size, input_size)
    // weight shape: (output_size, input_size)
    // bias shape: (output_size)
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int output_size = weight.size(0);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor output = torch::zeros({batch_size, output_size}, options);
    
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocks((output_size + BLOCK_SIZE - 1) / BLOCK_SIZE, (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    const float sqrt_2_over_pi = 0.7978845608028654f; // sqrt(2/π)
    const float c = 0.044715f;
    
    float* input_ptr = input.data_ptr<float>();
    float* weight_ptr = weight.data_ptr<float>();
    float* bias_ptr = bias.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    hipLaunchKernelGGL(
        linear_div_gelu_kernel, 
        blocks, 
        threads, 
        0, 
        0,
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        batch_size, input_size, output_size, divisor,
        sqrt_2_over_pi, c
    );
    
    return output;
}
""

# Compile the fused kernel
linear_div_gelu_op = load_inline(
    name="linear_div_gelu",
    cpp_sources=cpp_source,
    functions=["linear_div_gelu"],
    verbose=True,
    extra_cflags=["-O3"],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor
        self.linear_div_gelu_op = linear_div_gelu_op
        
    def forward(self, x):
        # Get the weight and bias from the linear layer
        weight = self.linear.weight
        bias = self.linear.bias
        
        # Call the fused kernel
        return self.linear_div_gelu_op.linear_div_gelu(x, weight, bias, self.divisor)


def get_inputs():
    return [torch.rand(1024, 8192)]


def get_init_inputs():
    return [8192, 8192, 10.0]
