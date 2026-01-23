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

__global__ void depthwise_conv2d_optimized_kernel(
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
    // Coalesced memory access pattern: have consecutive threads access contiguous memory
    
    const int batch = blockIdx.z / in_channels;
    const int channel = blockIdx.z % in_channels;
    const int out_row = blockIdx.y * TILE_HEIGHT + threadIdx.y;
    const int out_col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    if (out_row >= height_out || out_col >= width_out) return;
    
    // Calculate input coordinates (cache locally)
    const int in_row_start = out_row * stride - padding;
    const int in_col_start = out_col * stride - padding;
    
    // Load base pointers (use registers)
    const float* input_ptr = input + ((batch * in_channels + channel) * height * width);
    const float* weight_ptr = weight + (channel * KERNEL_SIZE * KERNEL_SIZE);
    float* output_ptr = output + ((batch * in_channels + channel) * height_out * width_out);
    
    float sum = 0.0f;
    
    // Precompute row boundaries
    const int in_row_valid_min = max(0, in_row_start);
    const int in_row_valid_max = min(height - 1, in_row_start + KERNEL_SIZE - 1);
    
    const int in_col_valid_min = max(0, in_col_start);
    const int in_col_valid_max = min(width - 1, in_col_col_start + KERNEL_SIZE - 1);
    
    // Tight loop with minimized divergence and no if statements inside kernel loop
    // We compute kernel weights for all positions, multiply by zero if out of bounds
    #pragma unroll
    for (int kh = 0; kh < KERNEL_SIZE; kh++) {
        const int in_row = in_row_start + kh;
        
        // Check bounds once per row
        const bool row_valid = (in_row >= 0 && in_row < height);
        
        if (row_valid) {
            #pragma unroll
            for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                const int in_col = in_col_start + kw;
                
                // For each column, check bounds and compute
                const bool valid = (in_col >= 0 && in_col < width);
                const float input_val = valid ? input_ptr[in_row * width + in_col] : 0.0f;
                const float weight_val = weight_ptr[kh * KERNEL_SIZE + kw];
                sum += input_val * weight_val;
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
    
    // Optimize grid dimensions to reduce wasted threads
    const dim3 block(TILE_WIDTH, TILE_HEIGHT, 1);
    const dim3 grid(
        (width_out + TILE_WIDTH - 1) / TILE_WIDTH,
        (height_out + TILE_HEIGHT - 1) / TILE_HEIGHT,
        batch_size * in_channels
    );
    
    depthwise_conv2d_optimized_kernel<<<grid, block>>>(
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