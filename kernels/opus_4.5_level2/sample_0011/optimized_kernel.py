import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel: Scale + MaxPool2d + Clamp
fused_scale_maxpool_clamp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <float.h>

__global__ void fused_scale_maxpool_clamp_kernel(
    const float* __restrict__ input,
    const float* __restrict__ scale,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int pool_size,
    const float clamp_min,
    const float clamp_max
) {
    // Each thread computes one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_height * out_width;
    
    if (idx >= total) return;
    
    // Decode index
    int ow = idx % out_width;
    int oh = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int b = idx / (out_width * out_height * channels);
    
    // Get scale for this channel
    float s = scale[c];
    
    // Compute max over pooling window
    float max_val = -FLT_MAX;
    
    int h_start = oh * pool_size;
    int w_start = ow * pool_size;
    int h_end = min(h_start + pool_size, in_height);
    int w_end = min(w_start + pool_size, in_width);
    
    for (int h = h_start; h < h_end; h++) {
        for (int w = w_start; w < w_end; w++) {
            int in_idx = ((b * channels + c) * in_height + h) * in_width + w;
            float val = input[in_idx] * s;  // Apply scale
            max_val = fmaxf(max_val, val);
        }
    }
    
    // Clamp
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
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    
    const int out_height = in_height / pool_size;
    const int out_width = in_width / pool_size;
    
    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());
    
    const int total = batch_size * channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    
    fused_scale_maxpool_clamp_kernel<<<num_blocks, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        scale.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_height,
        in_width,
        out_height,
        out_width,
        pool_size,
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
    name="fused_scale_maxpool_clamp",
    cpp_sources=fused_scale_maxpool_clamp_cpp,
    cuda_sources=fused_scale_maxpool_clamp_source,
    functions=["fused_scale_maxpool_clamp_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized model with fused Scale + MaxPool + Clamp kernel.
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
        # Fused Scale + MaxPool + Clamp
        x = self.fused_module.fused_scale_maxpool_clamp_hip(
            x.contiguous(),
            self.scale.view(-1).contiguous(),
            self.maxpool_kernel_size,
            self.clamp_min,
            self.clamp_max
        )
        return x


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

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max]


def custom_kernel(inputs):
    model = ModelNew(*get_init_inputs()).cuda()
    return model(inputs[0])
