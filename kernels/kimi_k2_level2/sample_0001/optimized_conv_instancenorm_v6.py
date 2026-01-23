import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os
import math

os.environ["CXX"] = "hipcc"

# Define the combined HIP kernel source code
hip_source = """
#include <hip/hip_runtime.h>
#include <math.h>

#define BLOCK_SIZE 8

// Optimized Conv2d with bias kernel
__global__ void conv2d_bias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size,
    int out_height, int out_width
) {
    int b = blockIdx.x;
    int oc = blockIdx.y;
    int oh = blockIdx.z * BLOCK_SIZE + threadIdx.y;
    int ow = threadIdx.x;
    
    if (b >= batch_size || oc >= out_channels || oh >= out_height || ow >= out_width) return;
    
    float sum = bias[oc];  // Initialize with bias value
    
    // Optimized convolution with better memory access pattern
    int input_batch_offset = b * in_channels * height * width;
    int output_batch_offset = b * out_channels * out_height * out_width;
    int weight_oc_offset = oc * in_channels * kernel_size * kernel_size;
    
    for (int ic = 0; ic < in_channels; ++ic) {
        int input_ch_offset = input_batch_offset + ic * height * width;
        int weight_ch_offset = weight_oc_offset + ic * kernel_size * kernel_size;
        
        for (int kh = 0; kh < kernel_size; ++kh) {
            int ih = oh + kh;
            int input_h_offset = input_ch_offset + ih * width;
            int weight_h_offset = weight_ch_offset + kh * kernel_size;
            
            for (int kw = 0; kw < kernel_size; ++kw) {
                int iw = ow + kw;
                
                float input_val = input[input_h_offset + iw];
                float weight_val = weight[weight_h_offset + kw];
                sum += input_val * weight_val;
            }
        }
    }
    
    int output_offset = output_batch_offset + oc * out_height * out_width + oh * out_width + ow;
    output[output_offset] = sum;
}

// Fused InstanceNorm + Div kernel with parallel reduction
__global__ void instancenorm_div_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int batch_size, int channels, int height, int width,
    float eps, float divide_by
) {
    __shared__ float sdata[256];
    
    int b = blockIdx.x;
    int c = blockIdx.y;
    int tid = threadIdx.x;
    int num_elements = height * width;
    
    if (b >= batch_size || c >= channels) return;
    
    int offset = (b * channels + c) * num_elements;
    
    // Compute mean using parallel reduction
    float sum = 0.0f;
    for (int i = tid; i < num_elements; i += blockDim.x) {
        sum += input[offset + i];
    }
    
    // Parallel reduction
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    float mean = sdata[0] / num_elements;
    
    // Compute variance
    __syncthreads();
    float sum_sq = 0.0f;
    for (int i = tid; i < num_elements; i += blockDim.x) {
        float diff = input[offset + i] - mean;
        sum_sq += diff * diff;
    }
    
    sdata[tid] = sum_sq;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    float var = sdata[0] / num_elements;
    float inv_std = rsqrtf(var + eps);
    
    // Apply normalization and division
    for (int i = tid; i < num_elements; i += blockDim.x) {
        output[offset + i] = ((input[offset + i] - mean) * inv_std) / divide_by;
    }
}

// Wrapper functions
torch::Tensor conv2d_bias_hip(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    int out_height = height - kernel_size + 1;
    int out_width = width - kernel_size + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                               torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    int tile_h = (out_height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(batch_size, out_channels, tile_h);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    
    conv2d_bias_kernel<<<grid, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, out_channels, height, width, kernel_size, out_height, out_width
    );
    
    return output;
}

torch::Tensor instancenorm_div_hip(torch::Tensor input, float eps, float divide_by) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    dim3 block(256);
    dim3 grid(batch_size, channels);
    
    auto output = torch::zeros_like(input);
    
    instancenorm_div_kernel<<<grid, block>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        batch_size, channels, height, width, eps, divide_by
    );
    
    return output;
}
"""

# Compile the HIP kernels
conv_instancenorm_div = load_inline(
    name='conv_instancenorm_div_v6',
    cpp_sources=hip_source,
    functions=['conv2d_bias_hip', 'instancenorm_div_hip'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        # Store parameters
        self.divide_by = divide_by
        
        # Use actual PyTorch Conv2d layer to get identical weights/biases
        # but replace its forward method with our custom kernel
        self.register_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size))
        
        # Store reference to the compiled kernels
        self.conv_instancenorm_div = conv_instancenorm_div

    def forward(self, x):
        # Use custom kernel with PyTorch Conv2d's weights and bias
        x = self.conv_instancenorm_div.conv2d_bias_hip(
            x, self.conv.weight, self.conv.bias
        )
        
        # Fused InstanceNorm + Division
        x = self.conv_instancenorm_div.instancenorm_div_hip(x, 1e-5, self.divide_by)
        
        return x

def get_inputs():
    # Same as original
    batch_size = 128
    in_channels = 64
    height = width = 128
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    # Same as original
    in_channels = 64
    out_channels = 128
    kernel_size = 3
    divide_by = 2.0
    return [in_channels, out_channels, kernel_size, divide_by]