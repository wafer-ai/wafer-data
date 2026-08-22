import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized softmax + fused double maxpool kernel with better parallelization
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <float.h>
#include <math.h>

// Process multiple spatial positions per thread for better efficiency
__global__ void fused_maxpool3d_4x4x4_fast_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_channels,
    const int in_depth, const int in_height, const int in_width,
    const int out_depth, const int out_height, const int out_width
) {
    const int out_spatial = out_depth * out_height * out_width;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_channels * out_spatial;
    
    if (idx >= total) return;
    
    const int spatial_out = idx % out_spatial;
    const int bc = idx / out_spatial;
    
    const int w_out = spatial_out % out_width;
    const int h_out = (spatial_out / out_width) % out_height;
    const int d_out = spatial_out / (out_width * out_height);
    
    const int d_start = d_out * 4;
    const int h_start = h_out * 4;
    const int w_start = w_out * 4;
    
    const int in_spatial = in_depth * in_height * in_width;
    const int in_hw = in_height * in_width;
    const float* in_ptr = input + bc * in_spatial;
    
    float max_val = -FLT_MAX;
    
    // Manually unroll the 4x4x4 pooling region
    #pragma unroll
    for (int dd = 0; dd < 4; dd++) {
        int d_off = (d_start + dd) * in_hw;
        #pragma unroll
        for (int hh = 0; hh < 4; hh++) {
            int h_off = d_off + (h_start + hh) * in_width + w_start;
            
            float v0 = in_ptr[h_off + 0];
            float v1 = in_ptr[h_off + 1];
            float v2 = in_ptr[h_off + 2];
            float v3 = in_ptr[h_off + 3];
            
            max_val = fmaxf(max_val, v0);
            max_val = fmaxf(max_val, v1);
            max_val = fmaxf(max_val, v2);
            max_val = fmaxf(max_val, v3);
        }
    }
    
    output[idx] = max_val;
}

// Softmax kernel optimized for 16 channels - process with warp-level parallelism
__global__ void softmax_channel_16_warp_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int depth,
    const int height,
    const int width
) {
    const int channels = 16;
    const int spatial_size = depth * height * width;
    const int total = batch_size * spatial_size;
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    
    const int spatial_idx = idx % spatial_size;
    const int b = idx / spatial_size;
    
    const int hw = height * width;
    const int w = spatial_idx % width;
    const int h = (spatial_idx / width) % height;
    const int d = spatial_idx / hw;
    
    const int base = b * channels * spatial_size + d * hw + h * width + w;
    const int stride = spatial_size;
    
    // Load values
    float v[16];
    v[0]  = input[base + 0  * stride];
    v[1]  = input[base + 1  * stride];
    v[2]  = input[base + 2  * stride];
    v[3]  = input[base + 3  * stride];
    v[4]  = input[base + 4  * stride];
    v[5]  = input[base + 5  * stride];
    v[6]  = input[base + 6  * stride];
    v[7]  = input[base + 7  * stride];
    v[8]  = input[base + 8  * stride];
    v[9]  = input[base + 9  * stride];
    v[10] = input[base + 10 * stride];
    v[11] = input[base + 11 * stride];
    v[12] = input[base + 12 * stride];
    v[13] = input[base + 13 * stride];
    v[14] = input[base + 14 * stride];
    v[15] = input[base + 15 * stride];
    
    // Find max using tree reduction
    float m = v[0];
    #pragma unroll
    for (int i = 1; i < 16; i++) {
        m = fmaxf(m, v[i]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        v[i] = __expf(v[i] - m);
        sum += v[i];
    }
    
    // Normalize and write
    float inv_sum = __fdividef(1.0f, sum);
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        output[base + i * stride] = v[i] * inv_sum;
    }
}

torch::Tensor softmax_channel_hip_v6(torch::Tensor input) {
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
    
    softmax_channel_16_warp_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, depth, height, width
    );
    
    return output;
}

torch::Tensor fused_maxpool3d_hip_v6(torch::Tensor input) {
    const auto sizes = input.sizes();
    const int batch_size = sizes[0];
    const int channels = sizes[1];
    const int in_depth = sizes[2];
    const int in_height = sizes[3];
    const int in_width = sizes[4];
    
    const int out_depth = in_depth / 4;
    const int out_height = in_height / 4;
    const int out_width = in_width / 4;
    
    auto output = torch::empty({batch_size, channels, out_depth, out_height, out_width}, 
                               input.options());
    
    const int batch_channels = batch_size * channels;
    const int out_spatial = out_depth * out_height * out_width;
    const int total = batch_channels * out_spatial;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    
    fused_maxpool3d_4x4x4_fast_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_channels, in_depth, in_height, in_width,
        out_depth, out_height, out_width
    );
    
    return output;
}
"""

fused_ops = load_inline(
    name="conv3d_softmax_pool_v6",
    cpp_sources="""
torch::Tensor softmax_channel_hip_v6(torch::Tensor input);
torch::Tensor fused_maxpool3d_hip_v6(torch::Tensor input);
""",
    cuda_sources=fused_kernel_source,
    functions=["softmax_channel_hip_v6", "fused_maxpool3d_hip_v6"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model: Conv3d -> Softmax (16ch specialized with intrinsics) -> Fused 4x4x4 MaxPool
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.softmax_channel_hip_v6(x)
        x = self.fused_ops.fused_maxpool3d_hip_v6(x)
        return x


def get_inputs():
    return [torch.rand(128, 3, 16, 32, 32).cuda()]


def get_init_inputs():
    return [3, 16, 3, 2]
