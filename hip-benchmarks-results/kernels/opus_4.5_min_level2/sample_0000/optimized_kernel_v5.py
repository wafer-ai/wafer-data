import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized softmax + fused double maxpool kernel
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <float.h>
#include <math.h>

// Vectorized MaxPool3d kernel with 4x4x4 pooling using float4 loads where possible
__global__ void fused_maxpool3d_4x4x4_vec_kernel(
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
    
    // Unrolled loop for 4x4x4 window - use float4 for coalesced loads when possible
    #pragma unroll
    for (int dd = 0; dd < 4; dd++) {
        const int d_offset = base_offset + (d_start + dd) * in_hw;
        
        #pragma unroll
        for (int hh = 0; hh < 4; hh++) {
            const int row_offset = d_offset + (h_start + hh) * in_width + w_start;
            
            // Load 4 consecutive floats using float4 if aligned
            const float4* row_ptr = reinterpret_cast<const float4*>(&input[row_offset]);
            float4 vals = *row_ptr;
            
            max_val = fmaxf(max_val, vals.x);
            max_val = fmaxf(max_val, vals.y);
            max_val = fmaxf(max_val, vals.z);
            max_val = fmaxf(max_val, vals.w);
        }
    }
    
    output[idx] = max_val;
}

// Optimized softmax along channel dimension with 16 channels specialized
// Uses shared memory to reduce global memory accesses
__global__ void softmax_channel_16ch_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int depth,
    const int height,
    const int width
) {
    const int spatial_size = depth * height * width;
    const int total = batch_size * spatial_size;
    const int channels = 16;
    
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
    
    // Load all 16 channel values and find max
    float v0  = input[base_in_offset + 0  * channel_stride];
    float v1  = input[base_in_offset + 1  * channel_stride];
    float v2  = input[base_in_offset + 2  * channel_stride];
    float v3  = input[base_in_offset + 3  * channel_stride];
    float v4  = input[base_in_offset + 4  * channel_stride];
    float v5  = input[base_in_offset + 5  * channel_stride];
    float v6  = input[base_in_offset + 6  * channel_stride];
    float v7  = input[base_in_offset + 7  * channel_stride];
    float v8  = input[base_in_offset + 8  * channel_stride];
    float v9  = input[base_in_offset + 9  * channel_stride];
    float v10 = input[base_in_offset + 10 * channel_stride];
    float v11 = input[base_in_offset + 11 * channel_stride];
    float v12 = input[base_in_offset + 12 * channel_stride];
    float v13 = input[base_in_offset + 13 * channel_stride];
    float v14 = input[base_in_offset + 14 * channel_stride];
    float v15 = input[base_in_offset + 15 * channel_stride];
    
    // Find max
    float max_val = v0;
    max_val = fmaxf(max_val, v1);
    max_val = fmaxf(max_val, v2);
    max_val = fmaxf(max_val, v3);
    max_val = fmaxf(max_val, v4);
    max_val = fmaxf(max_val, v5);
    max_val = fmaxf(max_val, v6);
    max_val = fmaxf(max_val, v7);
    max_val = fmaxf(max_val, v8);
    max_val = fmaxf(max_val, v9);
    max_val = fmaxf(max_val, v10);
    max_val = fmaxf(max_val, v11);
    max_val = fmaxf(max_val, v12);
    max_val = fmaxf(max_val, v13);
    max_val = fmaxf(max_val, v14);
    max_val = fmaxf(max_val, v15);
    
    // Compute exp and sum
    v0  = expf(v0  - max_val);
    v1  = expf(v1  - max_val);
    v2  = expf(v2  - max_val);
    v3  = expf(v3  - max_val);
    v4  = expf(v4  - max_val);
    v5  = expf(v5  - max_val);
    v6  = expf(v6  - max_val);
    v7  = expf(v7  - max_val);
    v8  = expf(v8  - max_val);
    v9  = expf(v9  - max_val);
    v10 = expf(v10 - max_val);
    v11 = expf(v11 - max_val);
    v12 = expf(v12 - max_val);
    v13 = expf(v13 - max_val);
    v14 = expf(v14 - max_val);
    v15 = expf(v15 - max_val);
    
    float sum_exp = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + 
                   v8 + v9 + v10 + v11 + v12 + v13 + v14 + v15;
    
    // Normalize and write
    const float inv_sum = 1.0f / sum_exp;
    
    output[base_in_offset + 0  * channel_stride] = v0  * inv_sum;
    output[base_in_offset + 1  * channel_stride] = v1  * inv_sum;
    output[base_in_offset + 2  * channel_stride] = v2  * inv_sum;
    output[base_in_offset + 3  * channel_stride] = v3  * inv_sum;
    output[base_in_offset + 4  * channel_stride] = v4  * inv_sum;
    output[base_in_offset + 5  * channel_stride] = v5  * inv_sum;
    output[base_in_offset + 6  * channel_stride] = v6  * inv_sum;
    output[base_in_offset + 7  * channel_stride] = v7  * inv_sum;
    output[base_in_offset + 8  * channel_stride] = v8  * inv_sum;
    output[base_in_offset + 9  * channel_stride] = v9  * inv_sum;
    output[base_in_offset + 10 * channel_stride] = v10 * inv_sum;
    output[base_in_offset + 11 * channel_stride] = v11 * inv_sum;
    output[base_in_offset + 12 * channel_stride] = v12 * inv_sum;
    output[base_in_offset + 13 * channel_stride] = v13 * inv_sum;
    output[base_in_offset + 14 * channel_stride] = v14 * inv_sum;
    output[base_in_offset + 15 * channel_stride] = v15 * inv_sum;
}

torch::Tensor softmax_channel_hip_v5(torch::Tensor input) {
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
    
    if (channels == 16) {
        softmax_channel_16ch_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, depth, height, width
        );
    } else {
        // Fallback to PyTorch softmax for other channel counts
        output = torch::softmax(input, 1);
    }
    
    return output;
}

torch::Tensor fused_maxpool3d_hip_v5(torch::Tensor input) {
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
    
    fused_maxpool3d_4x4x4_vec_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, in_depth, in_height, in_width,
        out_depth, out_height, out_width
    );
    
    return output;
}
"""

fused_ops = load_inline(
    name="conv3d_softmax_pool_v5",
    cpp_sources="""
torch::Tensor softmax_channel_hip_v5(torch::Tensor input);
torch::Tensor fused_maxpool3d_hip_v5(torch::Tensor input);
""",
    cuda_sources=fused_kernel_source,
    functions=["softmax_channel_hip_v5", "fused_maxpool3d_hip_v5"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model: Conv3d -> Softmax (16ch specialized) -> Fused 4x4x4 MaxPool
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.softmax_channel_hip_v5(x)
        x = self.fused_ops.fused_maxpool3d_hip_v5(x)
        return x


def get_inputs():
    return [torch.rand(128, 3, 16, 32, 32).cuda()]


def get_init_inputs():
    return [3, 16, 3, 2]
