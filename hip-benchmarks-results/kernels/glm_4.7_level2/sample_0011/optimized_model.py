import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Simple and correct MaxPool + Clamp kernel
maxpool_clamp_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void maxpool_clamp_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    float clamp_min,
    float clamp_max) {
    
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_idx = blockIdx.z;
    
    if (out_x >= out_width || out_y >= out_height || batch_idx >= batch_size) return;
    
    // Process all channels for this batch
    for (int channel_idx = 0; channel_idx < channels; channel_idx++) {
        // Calculate input window
        int in_x_start = out_x * stride;
        int in_y_start = out_y * stride;
        
        // Find max value in kernel window
        float max_val = -1e38f;
        
        for (int ky = 0; ky < kernel_size; ky++) {
            int in_y = in_y_start + ky;
            if (in_y < 0 || in_y >= in_height) continue;
            
            for (int kx = 0; kx < kernel_size; kx++) {
                int in_x = in_x_start + kx;
                if (in_x < 0 || in_x >= in_width) continue;
                
                // Input index: [batch, channel, height, width]
                int idx = ((batch_idx * channels + channel_idx) * in_height + in_y) * in_width + in_x;
                float val = input[idx];
                
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
        
        // Apply clamp
        if (max_val < clamp_min) max_val = clamp_min;
        if (max_val > clamp_max) max_val = clamp_max;
        
        // Output index: [batch, channel, height, width]
        int out_idx = ((batch_idx * channels + channel_idx) * out_height + out_y) * out_width + out_x;
        output[out_idx] = max_val;
    }
}

torch::Tensor maxpool_clamp_hip(
    torch::Tensor input,
    int kernel_size,
    float clamp_min,
    float clamp_max) {
    
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    
    int stride = kernel_size;  // Default stride = kernel_size
    
    int out_height = (in_height - kernel_size) / stride + 1;
    int out_width = (in_width - kernel_size) / stride + 1;
    
    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());
    
    const int block_x = 16;
    const int block_y = 8;
    dim3 threads(block_x, block_y);
    
    dim3 blocks(
        (out_width + block_x - 1) / block_x,
        (out_height + block_y - 1) / block_y,
        batch_size
    );
    
    maxpool_clamp_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        stride,
        clamp_min,
        clamp_max);
    
    return output;
}
"""

maxpool_clamp = load_inline(
    name="maxpool_clamp",
    cpp_sources=maxpool_clamp_cpp_source,
    functions=["maxpool_clamp_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model with fused custom HIP kernel for MaxPool+Clamp.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        # Keep Conv2d (highly optimized in cuDNN)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        
        # Keep standard GroupNorm
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        
        # Scale parameter
        self.scale = nn.Parameter(torch.ones(scale_shape))
        
        # Replace MaxPool2d + Clamp with fused kernel
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        
        # Load custom kernel
        self.maxpool_clamp_fn = maxpool_clamp

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            Output tensor of shape (batch_size, out_channels, height', width').
        """
        # Step 1: Conv2d
        x = self.conv(x)
        
        # Step 2: GroupNorm (keep standard)
        x = self.group_norm(x)
        
        # Step 3: Scale
        x = x * self.scale
        
        # Step 4: Fused MaxPool + Clamp
        x = self.maxpool_clamp_fn.maxpool_clamp_hip(
            x,
            self.maxpool_kernel_size,
            self.clamp_min,
            self.clamp_max
        )
        
        return x