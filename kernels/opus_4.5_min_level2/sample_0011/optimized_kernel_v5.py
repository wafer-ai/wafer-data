import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel with higher ILP - each thread processes 4 output elements
fused_scale_maxpool_clamp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <float.h>

// Kernel where each thread processes 4 consecutive output pixels horizontally
__global__ void fused_scale_maxpool_clamp_kernel_v5(
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
    // Each thread processes 4 consecutive output elements in width dimension
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_hw_4 = (out_height * out_width + 3) / 4;  // Number of 4-element groups
    int total_groups = batch_size * channels * out_hw_4;
    
    if (idx >= total_groups) return;
    
    int group_in_plane = idx % out_hw_4;
    int c = (idx / out_hw_4) % channels;
    int b = idx / (out_hw_4 * channels);
    
    // Get scale value
    float scale_val = scale[c];
    
    // Base offset in input
    int in_plane_size = in_height * in_width;
    int out_plane_size = out_height * out_width;
    int in_base = b * channels * in_plane_size + c * in_plane_size;
    int out_base = b * channels * out_plane_size + c * out_plane_size;
    
    // Process up to 4 consecutive output elements
    int base_out_idx = group_in_plane * 4;
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int out_flat = base_out_idx + i;
        if (out_flat >= out_plane_size) break;
        
        int ow = out_flat % out_width;
        int oh = out_flat / out_width;
        
        int h_start = oh * 4;
        int w_start = ow * 4;
        
        float max_val = -FLT_MAX;
        
        // 4x4 pooling window
        #pragma unroll
        for (int ph = 0; ph < 4; ph++) {
            int row_offset = in_base + (h_start + ph) * in_width + w_start;
            #pragma unroll
            for (int pw = 0; pw < 4; pw++) {
                float val = input[row_offset + pw] * scale_val;
                max_val = fmaxf(max_val, val);
            }
        }
        
        // Clamp
        max_val = fminf(fmaxf(max_val, clamp_min), clamp_max);
        output[out_base + out_flat] = max_val;
    }
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
    
    int out_hw_4 = (out_height * out_width + 3) / 4;
    int total_groups = batch_size * channels * out_hw_4;
    
    const int block_size = 256;
    int num_blocks = (total_groups + block_size - 1) / block_size;
    
    fused_scale_maxpool_clamp_kernel_v5<<<num_blocks, block_size>>>(
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
    name="fused_scale_maxpool_clamp_v5",
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
