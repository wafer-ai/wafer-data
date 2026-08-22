import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

maxpool2d_cpp_source = """
#include <hip/hip_runtime.h>
#include <math.h>

__constant__ const int KERNEL_SIZE_C = 4;

template<typename T>
__global__ void maxpool2d_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    int output_x = blockIdx.x * blockDim.x + threadIdx.x;
    int output_y = blockIdx.y * blockDim.y + threadIdx.y;
    int linear_z = blockIdx.z * blockDim.z + threadIdx.z;
    
    int b = linear_z / channels;
    int c = linear_z % channels;
    
    if (output_x >= output_width || output_y >= output_height || b >= batch_size) {
        return;
    }
    
    const int input_start_x = output_x * stride - padding;
    const int input_start_y = output_y * stride - padding;
    const int kernel_radius_x = dilation * (kernel_size - 1);
    const int kernel_radius_y = dilation * (kernel_size - 1);
    
    // Compute valid region in input space
    const int input_x_min = max(0, input_start_x);
    const int input_x_max = min(input_width - 1, input_start_x + kernel_radius_x);
    const int input_y_min = max(0, input_start_y);
    const int input_y_max = min(input_height - 1, input_start_y + kernel_radius_y);
    
    T max_val = -INFINITY;
    
    // Precompute base offset
    const int batch_channel_offset = b * channels * input_height * input_width 
                                   + c * input_height * input_width;
    
    // Loop through valid input region
    for (int input_y = input_y_min; input_y <= input_y_max; input_y += dilation) {
        const int row_offset = batch_channel_offset + input_y * input_width;
        
        for (int kx = 0; kx < kernel_size; ++kx) {
            int input_x = input_start_x + kx * dilation;
            if (input_x >= input_x_min && input_x <= input_x_max) {
                T val = input[row_offset + input_x];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }
    
    // Write output
    const int output_idx = b * channels * output_height * output_width 
                          + c * output_height * output_width 
                          + output_y * output_width 
                          + output_x;
    output[output_idx] = max_val;
}

torch::Tensor maxpool2d_hip(torch::Tensor input, int kernel_size, int stride, int padding, int dilation) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto input_height = input.size(2);
    auto input_width = input.size(3);
    
    // Calculate output dimensions
    int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, channels, output_height, output_width}, input.options());
    
    const int BLOCK_X = 32;
    const int BLOCK_Y = 8;
    const int BLOCK_Z = 2;
    
    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    
    dim3 grid;
    grid.x = (output_width + BLOCK_X - 1) / BLOCK_X;
    grid.y = (output_height + BLOCK_Y - 1) / BLOCK_Y;
    grid.z = (batch_size * channels + BLOCK_Z - 1) / BLOCK_Z;
    
    maxpool2d_kernel<float><<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        dilation
    );
    
    return output;
}
"""

maxpool2d = load_inline(
    name="maxpool2d",
    cpp_sources=maxpool2d_cpp_source,
    functions=["maxpool2d_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Simple model that performs Max Pooling 2D with optimized HIP kernel.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the Max Pooling 2D layer with HIP kernel.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.maxpool2d = maxpool2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 2D to the input tensor using HIP kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D, shape (batch_size, channels, pooled_height, pooled_width).
        """
        return self.maxpool2d.maxpool2d_hip(x, self.kernel_size, self.stride, self.padding, self.dilation)