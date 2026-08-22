import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel with tuned block sizes and improved memory access
fused_scale_maxpool_clamp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <float.h>

// Simple, efficient 1D kernel with 512-thread blocks for better occupancy
__global__ __launch_bounds__(512) void fused_scale_maxpool_clamp_kernel_v6(
    const float* __restrict__ input,
    const float* __restrict__ scale,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    float clamp_min,
    float clamp_max
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_height * out_width;
    
    if (idx >= total) return;
    
    // Decode index
    int ow = idx % out_width;
    int oh = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int b = idx / (out_width * out_height * channels);
    
    // Get scale value
    float scale_val = scale[c];
    
    // Compute max pooling with scale and clamp
    float max_val = -FLT_MAX;
    
    int h_start = oh << 2;  // oh * 4
    int w_start = ow << 2;  // ow * 4
    
    // Base offset for this batch and channel
    int base_offset = b * (channels * in_height * in_width) + c * (in_height * in_width);
    
    // Fully unrolled 4x4 pooling
    int row0 = base_offset + h_start * in_width + w_start;
    int row1 = row0 + in_width;
    int row2 = row1 + in_width;
    int row3 = row2 + in_width;
    
    // Row 0
    float v0 = input[row0] * scale_val;
    float v1 = input[row0 + 1] * scale_val;
    float v2 = input[row0 + 2] * scale_val;
    float v3 = input[row0 + 3] * scale_val;
    max_val = fmaxf(max_val, v0);
    max_val = fmaxf(max_val, v1);
    max_val = fmaxf(max_val, v2);
    max_val = fmaxf(max_val, v3);
    
    // Row 1
    v0 = input[row1] * scale_val;
    v1 = input[row1 + 1] * scale_val;
    v2 = input[row1 + 2] * scale_val;
    v3 = input[row1 + 3] * scale_val;
    max_val = fmaxf(max_val, v0);
    max_val = fmaxf(max_val, v1);
    max_val = fmaxf(max_val, v2);
    max_val = fmaxf(max_val, v3);
    
    // Row 2
    v0 = input[row2] * scale_val;
    v1 = input[row2 + 1] * scale_val;
    v2 = input[row2 + 2] * scale_val;
    v3 = input[row2 + 3] * scale_val;
    max_val = fmaxf(max_val, v0);
    max_val = fmaxf(max_val, v1);
    max_val = fmaxf(max_val, v2);
    max_val = fmaxf(max_val, v3);
    
    // Row 3
    v0 = input[row3] * scale_val;
    v1 = input[row3 + 1] * scale_val;
    v2 = input[row3 + 2] * scale_val;
    v3 = input[row3 + 3] * scale_val;
    max_val = fmaxf(max_val, v0);
    max_val = fmaxf(max_val, v1);
    max_val = fmaxf(max_val, v2);
    max_val = fmaxf(max_val, v3);
    
    // Clamp the result
    max_val = fminf(fmaxf(max_val, clamp_min), clamp_max);
    
    output[idx] = max_val;
}

torch::Tensor fused_scale_maxpool_clamp_hip(
    torch::Tensor input,
    torch::Tensor scale,
    int pool_size,
    float clamp_min,
    float clamp_max
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_height = in_height / pool_size;
    int out_width = in_width / pool_size;
    
    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());
    
    int total = batch_size * channels * out_height * out_width;
    const int block_size = 512;  // Larger block for better occupancy
    int num_blocks = (total + block_size - 1) / block_size;
    
    fused_scale_maxpool_clamp_kernel_v6<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        scale.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_height,
        in_width,
        out_height,
        out_width,
        clamp_min,
        clamp_max
    );
    
    return output;
}
"""

fused_scale_maxpool_clamp_cpp = """
torch::Tensor fused_scale_maxpool_clamp_hip(
    torch::Tensor input,
    torch::Tensor scale,
    int pool_size,
    float clamp_min,
    float clamp_max
);
"""

fused_module = load_inline(
    name="fused_scale_maxpool_clamp_v6",
    cpp_sources=fused_scale_maxpool_clamp_cpp,
    cuda_sources=fused_scale_maxpool_clamp_source,
    functions=["fused_scale_maxpool_clamp_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses scale, max pooling, and clamping into a single kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.fused_module = fused_module

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            Output tensor of shape (batch_size, out_channels, height', width').
        """
        x = self.conv(x)
        x = self.group_norm(x)
        # Fused scale + maxpool + clamp
        x = self.fused_module.fused_scale_maxpool_clamp_hip(
            x, 
            self.scale.view(-1),
            self.maxpool_kernel_size,
            self.clamp_min,
            self.clamp_max
        )
        return x
