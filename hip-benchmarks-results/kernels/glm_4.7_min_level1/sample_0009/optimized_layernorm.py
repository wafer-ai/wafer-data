import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

layernorm_cpp_source = """
#include <hip/hip_runtime.h>

#define EPSILON 1e-5f

__global__ void layernorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias_param,
    float* __restrict__ output,
    int batch_size,
    int features,
    int dim1,
    int dim2,
    int normalized_size)
{
    int batch_idx = blockIdx.y;
    int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || elem_idx >= normalized_size) return;
    
    int batch_offset = batch_idx * normalized_size;
    int idx = batch_offset + elem_idx;
    
    // Load element
    float val = input[idx];
    
    // Compute mean
    __shared__ float shmem[1024];
    shmem[threadIdx.x] = val;
    __syncthreads();
    
    // Reduce sum
    float sum = 0.0f;
    for (int i = 0; i < blockDim.x; i++) {
        int actual_idx = i + blockIdx.x * blockDim.x;
        sum += (actual_idx < normalized_size) ? shmem[i] : 0.0f;
    }
    
    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset);
    }
    __syncthreads();
    
    float mean = sum / normalized_size;
    
    // Compute variance
    float diff = val - mean;
    shmem[threadIdx.x] = diff * diff;
    __syncthreads();
    
    // Reduce var sum
    float var_sum = 0.0f;
    for (int i = 0; i < blockDim.x; i++) {
        int actual_idx = i + blockIdx.x * blockDim.x;
        var_sum += (actual_idx < normalized_size) ? shmem[i] : 0.0f;
    }
    var_sum /= normalized_size;
    
    // Warp reduce final var_sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        var_sum += __shfl_down(var_sum, offset);
    }
    __syncthreads();
    
    float inv_std = rsqrtf(var_sum + EPSILON);
    
    // Apply normalization with weight and bias
    output[idx] = weight[elem_idx] * diff * inv_std + bias_param[elem_idx];
}

torch::Tensor layernorm_hip(torch::Tensor input, torch::Tensor weight, torch::Tensor bias_param) {
    auto output = torch::zeros_like(input);
    int batch_size = input.size(0);
    int features = input.size(1);
    int dim1 = input.size(2);
    int dim2 = input.size(3);
    int normalized_size = features * dim1 * dim2;
    
    int block_size = 256;
    int num_blocks = (normalized_size + block_size - 1) / block_size;
    
    dim3 grid(num_blocks, batch_size);
    
    layernorm_kernel<<<grid, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_param.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        features,
        dim1,
        dim2,
        normalized_size
    );
    
    return output;
}
"""

layernorm = load_inline(
    name="layernorm",
    cpp_sources=layernorm_cpp_source,
    functions=["layernorm_hip"],
)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.layernorm_module = layernorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layernorm_module.layernorm_hip(x, self.weight, self.bias)


batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [(features, dim1, dim2)]