import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

# Define the combined HIP kernel source code
hip_source = """
#include <hip/hip_runtime.h>

#define BLOCK_SIZE 16

// Conv2d with bias kernel
__global__ void conv2d_bias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size,
    int out_height, int out_width
) {
    int b_oc = blockIdx.z;
    int b = b_oc / out_channels;
    int oc = b_oc % out_channels;
    int oh = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int ow = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    if (b >= batch_size || oc >= out_channels || oh >= out_height || ow >= out_width) return;
    
    float sum = bias[oc];  // Initialize with bias value
    
    // Compute convolution
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            int ih = oh + kh;
            for (int kw = 0; kw < kernel_size; ++kw) {
                int iw = ow + kw;
                
                float input_val = input[((b * in_channels + ic) * height + ih) * width + iw];
                float weight_val = weight[((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw];
                sum += input_val * weight_val;
            }
        }
    }
    
    output[((b * out_channels + oc) * out_height + oh) * out_width + ow] = sum;
}

// Fused InstanceNorm + Div kernel using parallel reduction
__global__ void instancenorm_div_kernel(
    float* __restrict__ data,
    int batch_size, int channels, int height, int width,
    float eps, float divide_by
) {
    __shared__ float sdata[256];
    
    int b = blockIdx.x;
    int c = blockIdx.y;
    int tid = threadIdx.x;
    int num_elements = height * width;
    
    if (b >= batch_size || c >= channels) return;
    
    // Compute mean using parallel reduction
    float sum = 0.0f;
    for (int i = tid; i < num_elements; i += blockDim.x) {
        int h = i / width;
        int w = i % width;
        sum += data[((b * channels + c) * height + h) * width + w];
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // Parallel reduction to compute total sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    float mean = sdata[0] / num_elements;
    
    // Compute variance using parallel reduction
    __syncthreads();
    float sum_sq = 0.0f;
    for (int i = tid; i < num_elements; i += blockDim.x) {
        int h = i / width;
        int w = i % width;
        float val = data[((b * channels + c) * height + h) * width + w];
        float diff = val - mean;
        sum_sq += diff * diff;
    }
    
    sdata[tid] = sum_sq;
    __syncthreads();
    
    // Parallel reduction for sum of squares
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    float var = sdata[0] / num_elements;
    
    // Apply normalization and division by constant
    float factor = rsqrtf(var + eps) / divide_by;
    for (int i = tid; i < num_elements; i += blockDim.x) {
        int h = i / width;
        int w = i % width;
        float val = data[((b * channels + c) * height + h) * width + w];
        data[((b * channels + c) * height + h) * width + w] = (val - mean) * factor;
    }
}

// Wrapper functions

// Conv2d wrapper
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
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    int num_blocks_h = (out_height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_blocks_w = (out_width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(num_blocks_w, num_blocks_h, batch_size * out_channels);
    
    conv2d_bias_kernel<<<grid, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, out_channels, height, width, kernel_size, out_height, out_width
    );
    
    return output;
}

// InstanceNorm + Div wrapper
torch::Tensor instancenorm_div_hip(torch::Tensor input, float eps, float divide_by) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    dim3 block(256);  // 256 threads per block for efficient reduction
    dim3 grid(batch_size, channels);
    
    // Create output tensor (don't modify input in-place)
    auto output = input.clone();
    
    instancenorm_div_kernel<<<grid, block>>>(
        output.data_ptr<float>(), batch_size, channels, height, width, eps, divide_by
    );
    
    return output;
}
"""

# Compile the HIP kernels
conv_instancenorm_div = load_inline(
    name='conv_instancenorm_div',
    cpp_sources=hip_source,
    functions=['conv2d_bias_hip', 'instancenorm_div_hip'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        # Store parameters
        self.divide_by = divide_by
        
        # Create and register weight and bias parameters (matching PyTorch Conv2d init)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Initialize parameters similar to PyTorch default initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_channels * kernel_size * kernel_size
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
        # Store reference to the compiled kernels
        self.conv_instancenorm_div = conv_instancenorm_div

    def forward(self, x):
        # Conv2d with bias
        x = self.conv_instancenorm_div.conv2d_bias_hip(x, self.weight, self.bias)
        
        # Fused InstanceNorm + Division
        x = self.conv_instancenorm_div.instancenorm_div_hip(x, 1e-5, self.divide_by)
        
        return x

# Import math for initialization
import math

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