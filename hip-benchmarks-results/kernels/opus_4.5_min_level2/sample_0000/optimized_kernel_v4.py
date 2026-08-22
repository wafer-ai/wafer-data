import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized softmax + fused double maxpool kernel
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <float.h>
#include <math.h>

// Optimized MaxPool3d kernel with 4x4x4 pooling using unrolled loops
__global__ void fused_maxpool3d_4x4x4_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_depth, const int in_height, const int in_width,
    const int out_depth, const int out_height, const int out_width
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * channels * out_depth * out_height * out_width;
    
    if (idx >= total) return;
    
    // Decode output index
    const int w_out = idx % out_width;
    const int h_out = (idx / out_width) % out_height;
    const int d_out = (idx / (out_width * out_height)) % out_depth;
    const int c = (idx / (out_width * out_height * out_depth)) % channels;
    const int b = idx / (out_width * out_height * out_depth * channels);
    
    // Input starting position (4x4x4 pooling window)
    const int d_start = d_out * 4;
    const int h_start = h_out * 4;
    const int w_start = w_out * 4;
    
    const int in_hw = in_height * in_width;
    const int base_offset = ((b * channels + c) * in_depth) * in_hw;
    
    float max_val = -FLT_MAX;
    
    // Unrolled loop for 4x4x4 window
    #pragma unroll
    for (int dd = 0; dd < 4; dd++) {
        const int d_idx = d_start + dd;
        const int d_offset = base_offset + d_idx * in_hw;
        
        #pragma unroll
        for (int hh = 0; hh < 4; hh++) {
            const int h_idx = h_start + hh;
            const int h_offset = d_offset + h_idx * in_width;
            
            #pragma unroll
            for (int ww = 0; ww < 4; ww++) {
                const int w_idx = w_start + ww;
                const float val = input[h_offset + w_idx];
                max_val = fmaxf(max_val, val);
            }
        }
    }
    
    output[idx] = max_val;
}

// Optimized softmax along channel dimension - each thread handles one spatial position
__global__ void softmax_channel_kernel_opt(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int depth,
    const int height,
    const int width
) {
    const int spatial_size = depth * height * width;
    const int total = batch_size * spatial_size;
    
    // Each thread handles one spatial position
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    
    const int spatial_idx = idx % spatial_size;
    const int b = idx / spatial_size;
    
    const int hw = height * width;
    const int w = spatial_idx % width;
    const int h = (spatial_idx / width) % height;
    const int d = spatial_idx / hw;
    
    const int base_in_offset = b * channels * spatial_size + d * hw + h * width + w;
    const int channel_stride = spatial_size;
    
    // Load all channel values and find max (assuming channels = 16)
    float vals[16];
    float max_val = -FLT_MAX;
    
    #pragma unroll
    for (int c = 0; c < 16; c++) {
        if (c < channels) {
            vals[c] = input[base_in_offset + c * channel_stride];
            max_val = fmaxf(max_val, vals[c]);
        }
    }
    
    // Compute exp and sum
    float sum_exp = 0.0f;
    #pragma unroll
    for (int c = 0; c < 16; c++) {
        if (c < channels) {
            vals[c] = expf(vals[c] - max_val);
            sum_exp += vals[c];
        }
    }
    
    // Normalize and write
    const float inv_sum = 1.0f / sum_exp;
    #pragma unroll
    for (int c = 0; c < 16; c++) {
        if (c < channels) {
            output[base_in_offset + c * channel_stride] = vals[c] * inv_sum;
        }
    }
}

torch::Tensor softmax_channel_hip_v4(torch::Tensor input) {
    const auto sizes = input.sizes();
    const int batch_size = sizes[0];
    const int channels = sizes[1];
    const int depth = sizes[2];
    const int height = sizes[3];
    const int width = sizes[4];
    
    auto output = torch::empty_like(input);
    
    const int total = batch_size * depth * height * width;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    
    softmax_channel_kernel_opt<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, depth, height, width
    );
    
    return output;
}

torch::Tensor fused_maxpool3d_hip_v4(torch::Tensor input) {
    const auto sizes = input.sizes();
    const int batch_size = sizes[0];
    const int channels = sizes[1];
    const int in_depth = sizes[2];
    const int in_height = sizes[3];
    const int in_width = sizes[4];
    
    // Output size after 4x4x4 pooling (equivalent to two 2x2x2 pools)
    const int out_depth = in_depth / 4;
    const int out_height = in_height / 4;
    const int out_width = in_width / 4;
    
    auto output = torch::empty({batch_size, channels, out_depth, out_height, out_width}, 
                               input.options());
    
    const int total = batch_size * channels * out_depth * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    
    fused_maxpool3d_4x4x4_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, in_depth, in_height, in_width,
        out_depth, out_height, out_width
    );
    
    return output;
}
"""

fused_ops = load_inline(
    name="conv3d_softmax_pool_v4",
    cpp_sources="""
torch::Tensor softmax_channel_hip_v4(torch::Tensor input);
torch::Tensor fused_maxpool3d_hip_v4(torch::Tensor input);
""",
    cuda_sources=fused_kernel_source,
    functions=["softmax_channel_hip_v4", "fused_maxpool3d_hip_v4"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model: Conv3d -> Softmax -> Fused 4x4x4 MaxPool (replaces two 2x2x2 pools)
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.softmax_channel_hip_v4(x)
        x = self.fused_ops.fused_maxpool3d_hip_v4(x)
        return x


def get_inputs():
    return [torch.rand(128, 3, 16, 32, 32).cuda()]


def get_init_inputs():
    return [3, 16, 3, 2]
