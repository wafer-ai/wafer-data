import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel: Tanh + Scaling + Bias Addition
fused_tanh_scale_bias_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void fused_tanh_scale_bias_kernel(
    const float* input, 
    const float* bias, 
    float* output, 
    int total_elements,
    int channels,
    int height,
    int width,
    float scaling_factor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        int channel_idx = (idx / (height * width)) % channels;
        
        // Tanh activation, scaling, and bias addition fused
        float val = tanhf(input[idx]);
        val = val * scaling_factor;
        output[idx] = val + bias[channel_idx];
    }
}

torch::Tensor fused_tanh_scale_bias_hip(
    torch::Tensor input, 
    torch::Tensor bias, 
    float scaling_factor
) {
    int total_elements = input.numel();
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    auto output = torch::zeros_like(input);
    
    const int block_size = 512;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_tanh_scale_bias_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        channels,
        height,
        width,
        scaling_factor
    );
    
    return output;
}
"""

# Simplified MaxPool kernel
maxpool_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void maxpool_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int pool_size
) {
    int batch_idx = blockIdx.z;
    int channel_idx = blockIdx.y;
    int out_pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out_pixels = out_height * out_width;
    
    if (batch_idx >= batch_size || channel_idx >= channels || out_pixel_idx >= total_out_pixels) {
        return;
    }
    
    int out_h = out_pixel_idx / out_width;
    int out_w = out_pixel_idx % out_width;
    
    int start_h = out_h * pool_size;
    int start_w = out_w * pool_size;
    
    float max_val = -1e30f;
    
    for (int ph = 0; ph < pool_size; ph++) {
        for (int pw = 0; pw < pool_size; pw++) {
            int in_h = start_h + ph;
            int in_w = start_w + pw;
            
            if (in_h < in_height && in_w < in_width) {
                int input_idx = ((batch_idx * channels + channel_idx) * in_height + in_h) * in_width + in_w;
                max_val = fmaxf(max_val, input[input_idx]);
            }
        }
    }
    
    int output_idx = ((batch_idx * channels + channel_idx) * out_height + out_h) * out_width + out_w;
    output[output_idx] = max_val;
}

torch::Tensor maxpool_hip(
    torch::Tensor input,
    int pool_size
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_height = in_height / pool_size;
    int out_width = in_width / pool_size;
    
    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());
    
    const int block_size = 256;
    int total_out_pixels = out_height * out_width;
    int num_blocks_x = (total_out_pixels + block_size - 1) / block_size;
    dim3 grid(num_blocks_x, channels, batch_size);
    
    maxpool_kernel<<<grid, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_height,
        in_width,
        out_height,
        out_width,
        pool_size
    );
    
    return output;
}
"""

fused_tanh_scale_bias = load_inline(
    name="fused_tanh_scale_bias",
    cpp_sources=fused_tanh_scale_bias_cpp_source,
    functions=["fused_tanh_scale_bias_hip"],
    verbose=True,
)

maxpool = load_inline(
    name="maxpool",
    cpp_sources=maxpool_cpp_source,
    functions=["maxpool_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model with fused HIP kernels.
    Uses PyTorch's optimized Conv2d but fuses Tanh+Scaling+BiasAdd and custom MaxPool.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.register_parameter('bias', nn.Parameter(torch.randn(bias_shape)))
        self.pool_kernel_size = pool_kernel_size
        
        # Load custom kernels
        self.fused_tanh_scale_bias = fused_tanh_scale_bias
        self.maxpool = maxpool

    def forward(self, x):
        # Convolution (use PyTorch's optimized implementation)
        x = self.conv(x)
        
        # Fused: Tanh + Scaling + Bias Addition
        x = self.fused_tanh_scale_bias.fused_tanh_scale_bias_hip(x, self.bias, self.scaling_factor)
        
        # Max-pooling with custom kernel
        x = self.maxpool.maxpool_hip(x, self.pool_kernel_size)
        
        return x