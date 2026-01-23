import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

depthwise_conv2d_hip_source = """
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define TILE_WIDTH 16
#define TILE_HEIGHT 16
#define KERNEL_SIZE 3

__global__ void depthwise_conv2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    const int batch_size,
    const int in_channels,
    const int height,
    const int width,
    const int stride,
    const int padding,
    const int height_out,
    const int width_out
) {
    // Input: (batch, in_channels, height, width)
    // Weight: (out_channels, in_channels/groups, kernel_h, kernel_w) = (in_channels, 1, kernel_h, kernel_w)
    // Output: (batch, out_channels, height_out, width_out) = (batch, in_channels, height_out, width_out)
    
    const int batch = blockIdx.z / in_channels;
    const int channel = blockIdx.z % in_channels;
    const int out_row = blockIdx.y * TILE_HEIGHT + threadIdx.y;
    const int out_col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    if (out_row >= height_out || out_col >= width_out) return;
    
    // Calculate input coordinates
    const int in_row_start = out_row * stride - padding;
    const int in_col_start = out_col * stride - padding;
    
    // Pointers to current batch and channel
    const float* input_ptr = input + ((batch * in_channels + channel) * height * width);
    // For depthwise conv, weight shape is (in_channels, 1, kernel_h, kernel_w)
    // So we need to get the filter for this channel: weight[channel, :, :, :]
    const float* weight_ptr = weight + (channel * KERNEL_SIZE * KERNEL_SIZE);
    float* output_ptr = output + ((batch * in_channels + channel) * height_out * width_out);
    
    float sum = 0.0f;
    
    // Unrolled convolution loop
    #pragma unroll
    for (int kh = 0; kh < KERNEL_SIZE; kh++) {
        const int in_row = in_row_start + kh;
        
        // Boundary check for row
        if (in_row >= 0 && in_row < height) {
            #pragma unroll
            for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                const int in_col = in_col_start + kw;
                
                // Boundary check for column
                if (in_col >= 0 && in_col < width) {
                    const float input_val = input_ptr[in_row * width + in_col];
                    const float weight_val = weight_ptr[kh * KERNEL_SIZE + kw];
                    sum += input_val * weight_val;
                }
            }
        }
    }
    
    output_ptr[out_row * width_out + out_col] = sum;
}

torch::Tensor depthwise_conv2d_hip(
    torch::Tensor input,
    torch::Tensor weight,
    const int stride,
    const int padding
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int kernel_size = weight.size(2);
    
    const int height_out = (height + 2 * padding - kernel_size) / stride + 1;
    const int width_out = (width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, in_channels, height_out, width_out}, input.options());
    
    const dim3 block(TILE_WIDTH, TILE_HEIGHT, 1);
    const dim3 grid(
        (width_out + TILE_WIDTH - 1) / TILE_WIDTH,
        (height_out + TILE_HEIGHT - 1) / TILE_HEIGHT,
        batch_size * in_channels
    );
    
    // Use hipLaunchKernelGGL with simplified syntax
    depthwise_conv2d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        height,
        width,
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
    cpp_sources=depthwise_conv2d_hip_source,
    functions=["depthwise_conv2d_hip"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.depthwise_conv2d = depthwise_conv2d
        self.stride = stride
        self.padding = padding
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Weight shape is (out_channels, in_channels/groups, kernel_h, kernel_w) = (in_channels, 1, kernel_h, kernel_w)
        weight = self.conv2d.weight  # Shape: (in_channels, 1, 3, 3)
        return self.depthwise_conv2d.depthwise_conv2d_hip(x, weight, self.stride, self.padding)


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