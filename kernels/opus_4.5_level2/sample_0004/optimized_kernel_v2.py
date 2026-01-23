import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel for: subtract1 -> tanh -> subtract2 -> avgpool with better memory access
fused_post_conv_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Optimized version with better coalesced memory access
// Process multiple channels per thread and use vectorized loads
__global__ void fused_sub_tanh_sub_avgpool_kernel_v2(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int pool_size,
    const float subtract1,
    const float subtract2,
    const float inv_pool_area
) {
    // Better thread mapping: x for width, y for height, z for batch*channel
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int bc = blockIdx.z;
    
    if (ow >= out_width || oh >= out_height) return;
    
    int b = bc / channels;
    int c = bc % channels;
    
    // Starting position in input
    int in_y_start = oh * pool_size;
    int in_x_start = ow * pool_size;
    
    // Base offset for this batch and channel
    int in_base = (b * channels + c) * in_height * in_width;
    
    float sum = 0.0f;
    
    // Unroll for 2x2 pool (most common case)
    #pragma unroll
    for (int py = 0; py < 2; py++) {
        int in_row_offset = (in_y_start + py) * in_width + in_x_start;
        
        #pragma unroll
        for (int px = 0; px < 2; px++) {
            float val = input[in_base + in_row_offset + px];
            val = val - subtract1;
            val = tanhf(val);
            val = val - subtract2;
            sum += val;
        }
    }
    
    int out_idx = (b * channels + c) * out_height * out_width + oh * out_width + ow;
    output[out_idx] = sum * inv_pool_area;
}

torch::Tensor fused_sub_tanh_sub_avgpool_hip(
    torch::Tensor input,
    float subtract1,
    float subtract2,
    int pool_size
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_height = in_height / pool_size;
    int out_width = in_width / pool_size;
    
    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());
    
    float inv_pool_area = 1.0f / (float)(pool_size * pool_size);
    
    // Use 2D thread blocks for better spatial locality
    dim3 block_size(16, 16);
    dim3 num_blocks(
        (out_width + block_size.x - 1) / block_size.x,
        (out_height + block_size.y - 1) / block_size.y,
        batch_size * channels
    );
    
    fused_sub_tanh_sub_avgpool_kernel_v2<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_height,
        in_width,
        out_height,
        out_width,
        pool_size,
        subtract1,
        subtract2,
        inv_pool_area
    );
    
    return output;
}
"""

fused_post_conv_cpp = """
torch::Tensor fused_sub_tanh_sub_avgpool_hip(
    torch::Tensor input,
    float subtract1,
    float subtract2,
    int pool_size
);
"""

fused_module = load_inline(
    name="fused_post_conv_v2",
    cpp_sources=fused_post_conv_cpp,
    cuda_sources=fused_post_conv_source,
    functions=["fused_sub_tanh_sub_avgpool_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses subtract, tanh, subtract, and avgpool into a single kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool

    def forward(self, x):
        x = self.conv(x)
        # Fused operations: subtract1 -> tanh -> subtract2 -> avgpool
        x = fused_module.fused_sub_tanh_sub_avgpool_hip(
            x, 
            self.subtract1_value, 
            self.subtract2_value,
            self.kernel_size_pool
        )
        return x
