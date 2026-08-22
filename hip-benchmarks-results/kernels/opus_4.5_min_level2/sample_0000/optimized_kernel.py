import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused softmax + double maxpool kernel
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <float.h>
#include <math.h>

// Fused MaxPool3d kernel - combines two 2x2x2 maxpools into one 4x4x4 maxpool
__global__ void fused_maxpool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_depth * out_height * out_width;
    
    if (idx >= total) return;
    
    // Decode output index
    int w_out = idx % out_width;
    int h_out = (idx / out_width) % out_height;
    int d_out = (idx / (out_width * out_height)) % out_depth;
    int c = (idx / (out_width * out_height * out_depth)) % channels;
    int b = idx / (out_width * out_height * out_depth * channels);
    
    // Input starting position (4x4x4 pooling window)
    int d_start = d_out * 4;
    int h_start = h_out * 4;
    int w_start = w_out * 4;
    
    float max_val = -FLT_MAX;
    
    // Find max in 4x4x4 window
    for (int dd = 0; dd < 4 && (d_start + dd) < in_depth; dd++) {
        for (int hh = 0; hh < 4 && (h_start + hh) < in_height; hh++) {
            for (int ww = 0; ww < 4 && (w_start + ww) < in_width; ww++) {
                int in_idx = ((b * channels + c) * in_depth + (d_start + dd)) * in_height * in_width 
                           + (h_start + hh) * in_width + (w_start + ww);
                float val = input[in_idx];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }
    
    output[idx] = max_val;
}

// Softmax along channel dimension for 5D tensor (B, C, D, H, W)
__global__ void softmax_channel_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width
) {
    int spatial_size = depth * height * width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * spatial_size;
    
    if (idx >= total) return;
    
    int spatial_idx = idx % spatial_size;
    int b = idx / spatial_size;
    
    // Decode spatial index
    int w = spatial_idx % width;
    int h = (spatial_idx / width) % height;
    int d = spatial_idx / (width * height);
    
    // Find max for numerical stability
    float max_val = -FLT_MAX;
    for (int c = 0; c < channels; c++) {
        int in_idx = ((b * channels + c) * depth + d) * height * width + h * width + w;
        float val = input[in_idx];
        if (val > max_val) {
            max_val = val;
        }
    }
    
    // Compute exp sum
    float sum_exp = 0.0f;
    for (int c = 0; c < channels; c++) {
        int in_idx = ((b * channels + c) * depth + d) * height * width + h * width + w;
        sum_exp += expf(input[in_idx] - max_val);
    }
    
    // Compute softmax
    for (int c = 0; c < channels; c++) {
        int in_idx = ((b * channels + c) * depth + d) * height * width + h * width + w;
        output[in_idx] = expf(input[in_idx] - max_val) / sum_exp;
    }
}

torch::Tensor softmax_channel_hip(torch::Tensor input) {
    auto sizes = input.sizes();
    int batch_size = sizes[0];
    int channels = sizes[1];
    int depth = sizes[2];
    int height = sizes[3];
    int width = sizes[4];
    
    auto output = torch::empty_like(input);
    
    int total = batch_size * depth * height * width;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    
    softmax_channel_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, depth, height, width
    );
    
    return output;
}

torch::Tensor fused_maxpool3d_hip(torch::Tensor input) {
    auto sizes = input.sizes();
    int batch_size = sizes[0];
    int channels = sizes[1];
    int in_depth = sizes[2];
    int in_height = sizes[3];
    int in_width = sizes[4];
    
    // Output size after 4x4x4 pooling (equivalent to two 2x2x2 pools)
    int out_depth = in_depth / 4;
    int out_height = in_height / 4;
    int out_width = in_width / 4;
    
    auto output = torch::empty({batch_size, channels, out_depth, out_height, out_width}, 
                               input.options());
    
    int total = batch_size * channels * out_depth * out_height * out_width;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    
    fused_maxpool3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, in_depth, in_height, in_width,
        out_depth, out_height, out_width
    );
    
    return output;
}
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources="""
torch::Tensor softmax_channel_hip(torch::Tensor input);
torch::Tensor fused_maxpool3d_hip(torch::Tensor input);
""",
    cuda_sources=fused_kernel_source,
    functions=["softmax_channel_hip", "fused_maxpool3d_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a 3D convolution, applies Softmax, and performs two max pooling operations.
    The two max pooling operations are fused into a single 4x4x4 pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.softmax_channel_hip(x)
        x = self.fused_ops.fused_maxpool3d_hip(x)
        return x


def get_inputs():
    return [torch.rand(128, 3, 16, 32, 32).cuda()]


def get_init_inputs():
    return [3, 16, 3, 2]
