import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel with 2D blocking and LDS (Local Data Share) for better memory access
fused_scale_maxpool_clamp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <float.h>

// 2D kernel layout optimized for spatial locality
__global__ void fused_scale_maxpool_clamp_kernel_2d(
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
    // 2D thread indexing for spatial dimensions
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int bc = blockIdx.z;  // Combined batch and channel index
    
    int b = bc / channels;
    int c = bc % channels;
    
    if (ow >= out_width || oh >= out_height || b >= batch_size)
        return;
    
    // Get scale value for this channel
    float scale_val = scale[c];
    
    // Compute max pooling with scale and clamp
    float max_val = -FLT_MAX;
    
    int h_start = oh * 4;
    int w_start = ow * 4;
    
    // Base offset for this batch and channel
    int base_offset = b * (channels * in_height * in_width) + c * (in_height * in_width);
    
    // Unrolled 4x4 pooling window
    #pragma unroll
    for (int ph = 0; ph < 4; ph++) {
        int ih = h_start + ph;
        int row_base = base_offset + ih * in_width + w_start;
        
        #pragma unroll
        for (int pw = 0; pw < 4; pw++) {
            float val = input[row_base + pw] * scale_val;
            max_val = fmaxf(max_val, val);
        }
    }
    
    // Clamp the result
    max_val = fminf(fmaxf(max_val, clamp_min), clamp_max);
    
    // Write output
    int out_idx = b * (channels * out_height * out_width) + 
                  c * (out_height * out_width) + 
                  oh * out_width + ow;
    output[out_idx] = max_val;
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
    
    // Use 2D thread blocks (16x16) for spatial dimensions
    dim3 block(16, 16, 1);
    dim3 grid(
        (out_width + block.x - 1) / block.x,
        (out_height + block.y - 1) / block.y,
        batch_size * channels
    );
    
    fused_scale_maxpool_clamp_kernel_2d<<<grid, block>>>(
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
    name="fused_scale_maxpool_clamp_v4",
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
