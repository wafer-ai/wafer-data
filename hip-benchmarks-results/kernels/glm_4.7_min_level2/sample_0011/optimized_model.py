import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

scale_maxpool_clamp_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void scale_maxpool_clamp_kernel(
    const float* __restrict__ input,
    const float* __restrict__ scale,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int height_in,
    int width_in,
    int height_out,
    int width_out,
    int pool_size,
    float clamp_min,
    float clamp_max) {
    
    int batch_idx = blockIdx.z;
    int channel_idx = blockIdx.y;
    int linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int pool_size_sq = pool_size * pool_size;
    
    // Each thread processes one output spatial position
    for (int i = threadIdx.x; i < height_out * width_out; i += blockDim.x) {
        int out_h = i / width_out;
        int out_w = i % width_out;
        
        if (batch_idx < batch_size && channel_idx < channels) {
            float scale_val = scale[channel_idx];
            int in_h = out_h * pool_size;
            int in_w = out_w * pool_size;
            
            // Single base input index
            int in_base = batch_idx * channels * height_in * width_in +
                          channel_idx * height_in * width_in +
                          in_h * width_in + in_w;
            
            // Max pooling
            float max_val = -1e30f;
            
            // Check and compute max
            if (in_h + pool_size <= height_in && in_w + pool_size <= width_in) {
                // Full window case
                for (int j = 0; j < pool_size_sq; j++) {
                    int ph = j / pool_size;
                    int pw = j % pool_size;
                    int idx = in_base + ph * width_in + pw;
                    float val = input[idx] * scale_val;
                    if (val > max_val) max_val = val;
                }
            } else {
                // Boundary case
                for (int ph = 0; ph < pool_size; ph++) {
                    for (int pw = 0; pw < pool_size; pw++) {
                        if (in_h + ph < height_in && in_w + pw < width_in) {
                            int idx = in_base + ph * width_in + pw;
                            float val = input[idx] * scale_val;
                            if (val > max_val) max_val = val;
                        }
                    }
                }
            }
            
            // Clamp
            if (max_val < clamp_min) max_val = clamp_min;
            if (max_val > clamp_max) max_val = clamp_max;
            
            // Write
            out_h = i / width_out;
            out_w = i % width_out;
            int out_idx = batch_idx * channels * height_out * width_out +
                          channel_idx * height_out * width_out +
                          out_h * width_out + out_w;
            output[out_idx] = max_val;
        }
    }
}

torch::Tensor scale_maxpool_clamp_hip(
    torch::Tensor input,
    torch::Tensor scale,
    int pool_size,
    float clamp_min,
    float clamp_max) {
    
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height_in = input.size(2);
    auto width_in = input.size(3);
    
    auto height_out = height_in / pool_size;
    auto width_out = width_in / pool_size;
    int spatial_size = height_out * width_out;
    
    auto output = torch::zeros({batch_size, channels, height_out, width_out}, input.options());
    
    // More efficient: fewer blocks, more threads per block
    const int threads_per_block = 256;
    int num_blocks = (spatial_size + threads_per_block - 1) / threads_per_block;
    
    dim3 grid_dim(num_blocks, channels, batch_size);
    
    scale_maxpool_clamp_kernel<<<grid_dim, threads_per_block>>>(
        input.data_ptr<float>(),
        scale.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height_in,
        width_in,
        height_out,
        width_out,
        pool_size,
        clamp_min,
        clamp_max);
    
    return output;
}
"""

scale_maxpool_clamp = load_inline(
    name="scale_maxpool_clamp",
    cpp_sources=scale_maxpool_clamp_cpp_source,
    functions=["scale_maxpool_clamp_hip"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses scale + maxpool + clamp into a single HIP kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale_maxpool_clamp = scale_maxpool_clamp

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            Output tensor of shape (batch_size, out_channels, height', width').
        """
        x = self.conv(x)
        x = self.group_norm(x)
        
        # Fuse scale * maxpool * clamp into single kernel
        x = self.scale_maxpool_clamp.scale_maxpool_clamp_hip(
            x, self.scale, self.maxpool_kernel_size, self.clamp_min, self.clamp_max)
        
        return x


def get_inputs():
    batch_size = 128
    in_channels = 8
    height, width = 128, 128
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    batch_size = 128
    in_channels = 8
    out_channels = 64
    height, width = 128, 128
    kernel_size = 3
    num_groups = 16
    scale_shape = (out_channels, 1, 1)
    maxpool_kernel_size = 4
    clamp_min = 0.0
    clamp_max = 1.0
    return [in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max]