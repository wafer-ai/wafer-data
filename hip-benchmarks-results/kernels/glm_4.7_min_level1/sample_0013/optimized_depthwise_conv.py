import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

depthwise_conv2d_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int height,
    const int width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int height_out,
    const int width_out
) {
    // Linear indexing for output
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_out_pixels = batch_size * in_channels * height_out * width_out;
    
    if (out_idx >= total_out_pixels) return;
    
    // Compute output position from linear index
    const int out_w = out_idx % width_out;
    const int out_h_rem = out_idx / width_out;
    const int out_h = out_h_rem % height_out;
    const int ch_rem = out_h_rem / height_out;
    const int ch = ch_rem % in_channels;
    const int b = ch_rem / in_channels;
    
    // Accumulator for this output pixel
    float sum = 0.0f;
    
    // Convolution loop
    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            // Compute input position
            const int in_h = out_h * stride - padding + kh;
            const int in_w = out_w * stride - padding + kw;
            
            // Check bounds
            if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                // Input index
                const int in_idx = b * in_channels * height * width + ch * height * width + in_h * width + in_w;
                
                // Weight index (depthwise: one 2D kernel per channel)
                const int w_idx = ch * kernel_size * kernel_size + kh * kernel_size + kw;
                
                sum += input[in_idx] * weight[w_idx];
            }
        }
    }
    
    // Write output
    output[out_idx] = sum;
}

torch::Tensor depthwise_conv2d_hip(torch::Tensor input, torch::Tensor weight, int kernel_size, int stride, int padding) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    auto height_out = (height + 2 * padding - kernel_size) / stride + 1;
    auto width_out = (width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, in_channels, height_out, width_out}, input.options());
    
    const int total_out_pixels = batch_size * in_channels * height_out * width_out;
    const int block_size = 256;
    const int num_blocks = (total_out_pixels + block_size - 1) / block_size;
    
    depthwise_conv2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        height_out,
        width_out
    );
    
    return output;
}
"""

depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    functions=["depthwise_conv2d_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution operation with square input and square kernel using custom HIP kernel.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Create weight tensor manually (since we're using custom kernel)
        # The weight tensor shape is (in_channels, 1, kernel_size, kernel_size) for depthwise conv
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter('bias', None)
            
        self.depthwise_conv2d = depthwise_conv2d
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using custom HIP kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        # Ensure inputs are on the same device
        device = x.device
        self.weight.data = self.weight.data.to(device)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(device)
        
        # Reshape weight from (in_channels, 1, kernel_size, kernel_size) to (in_channels, kernel_size, kernel_size)
        weight_flat = self.weight.squeeze(1)
        
        # Call the custom HIP kernel
        output = self.depthwise_conv2d.depthwise_conv2d_hip(x, weight_flat, self.kernel_size, self.stride, self.padding)
        
        # Add bias if present
        if self.bias is not None:
            bias_view = self.bias.view(1, -1, 1, 1)
            output = output + bias_view
            
        return output


# Test code
batch_size = 16
in_channels = 64
kernel_size = 3
width = 512
height = 512
stride = 1
padding = 0

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, kernel_size, stride, padding]