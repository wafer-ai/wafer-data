import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

instance_norm_divide_cpp_source = """
#include <hip/hip_runtime.h>

template<int BLOCK_SIZE>
__global__ void instance_norm_divide_kernel(
    const float* input, 
    float* output,
    const float* weight, 
    const float* bias,
    int batch_size, 
    int channels, 
    int height, 
    int width,
    float divide_by,
    float eps
) {
    int n = blockIdx.y;  // batch
    int c = blockIdx.x;  // channel
    
    if (n >= batch_size || c >= channels) return;
    
    int hw = height * width;
    int base_idx = (n * channels + c) * hw;
    const float* in_ptr = input + base_idx;
    float* out_ptr = output + base_idx;
    
    // Shared memory for reduction
    __shared__ float sum_smem[BLOCK_SIZE];
    __shared__ float var_smem[BLOCK_SIZE];
    
    // === Compute mean ===
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hw; i += BLOCK_SIZE) {
        sum += in_ptr[i];
    }
    
    sum_smem[threadIdx.x] = sum;
    __syncthreads();
    
    // Block reduction
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum_smem[threadIdx.x] += sum_smem[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    float mean = sum_smem[0] / (float)hw;
    __syncthreads();
    
    // === Compute variance ===
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < hw; i += BLOCK_SIZE) {
        float diff = in_ptr[i] - mean;
        var_sum += diff * diff;
    }
    
    var_smem[threadIdx.x] = var_sum;
    __syncthreads();
    
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            var_smem[threadIdx.x] += var_smem[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    float inv_std = 1.0f / sqrtf(fmaxf(var_smem[0] / (float)hw, eps));
    __syncthreads();
    
    // === Get affine parameters and combine with divide ===
    float w = weight[c];
    float b = bias[c];
    
    // Combine multiplication/division: (weight * normalized) / divide_by
    float scale = w / divide_by;
    float shift = b / divide_by;
    
    // === Normalize and output ===
    for (int i = threadIdx.x; i < hw; i += BLOCK_SIZE) {
        out_ptr[i] = scale * (in_ptr[i] - mean) * inv_std + shift;
    }
}

torch::Tensor instance_norm_divide_hip(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float divide_by,
    float eps
) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    auto output = torch::zeros_like(input);
    
    // Use block size of 512 for better throughput
    const int block_size = 512;
    dim3 block(block_size);
    dim3 grid(channels, batch_size);
    
    instance_norm_divide_kernel<block_size><<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width,
        divide_by,
        eps
    );
    
    return output;
}
"""

instance_norm_divide = load_inline(
    name="instance_norm_divide",
    cpp_sources=instance_norm_divide_cpp_source,
    functions=["instance_norm_divide_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model with fused InstanceNorm2d + Divide operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.instance_norm_divide = instance_norm_divide
        self.divide_by = divide_by
        
        # Create InstanceNorm parameters that match nn.InstanceNorm2d
        self.weight = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv(x)
        
        # Fused InstanceNorm + Divide
        x = self.instance_norm_divide.instance_norm_divide_hip(
            x,
            self.weight,
            self.bias,
            self.divide_by,
            self.eps
        )
        
        return x