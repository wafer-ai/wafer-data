import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused kernel using 2D grid with better memory coalescing
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define POOL_SIZE 4

// Use 2D block organization for better spatial locality
// Block: 16x16 threads = 256 threads = 4 wavefronts
__global__ __launch_bounds__(256) void fused_tanh_scale_bias_maxpool_kernel_v5(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const float scaling_factor,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width
) {
    // Output coordinates
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int bc = blockIdx.z; // Combined batch and channel index
    
    if (ow >= out_width || oh >= out_height) return;
    
    int b = bc / channels;
    int c = bc % channels;
    
    float bias_val = bias[c];
    
    int h_start = oh * POOL_SIZE;
    int w_start = ow * POOL_SIZE;
    
    // Base index for this batch and channel
    int plane_stride = in_height * in_width;
    const float* input_plane = input + (b * channels + c) * plane_stride;
    
    // Compute max over 4x4 pooling window - fully unrolled
    float max_val = -INFINITY;
    
    // Row 0
    {
        const float* row = input_plane + h_start * in_width + w_start;
        float v0 = row[0];
        float v1 = row[1];
        float v2 = row[2];
        float v3 = row[3];
        v0 = tanhf(v0) * scaling_factor + bias_val;
        v1 = tanhf(v1) * scaling_factor + bias_val;
        v2 = tanhf(v2) * scaling_factor + bias_val;
        v3 = tanhf(v3) * scaling_factor + bias_val;
        max_val = fmaxf(max_val, fmaxf(fmaxf(v0, v1), fmaxf(v2, v3)));
    }
    
    // Row 1
    {
        const float* row = input_plane + (h_start + 1) * in_width + w_start;
        float v0 = row[0];
        float v1 = row[1];
        float v2 = row[2];
        float v3 = row[3];
        v0 = tanhf(v0) * scaling_factor + bias_val;
        v1 = tanhf(v1) * scaling_factor + bias_val;
        v2 = tanhf(v2) * scaling_factor + bias_val;
        v3 = tanhf(v3) * scaling_factor + bias_val;
        max_val = fmaxf(max_val, fmaxf(fmaxf(v0, v1), fmaxf(v2, v3)));
    }
    
    // Row 2
    {
        const float* row = input_plane + (h_start + 2) * in_width + w_start;
        float v0 = row[0];
        float v1 = row[1];
        float v2 = row[2];
        float v3 = row[3];
        v0 = tanhf(v0) * scaling_factor + bias_val;
        v1 = tanhf(v1) * scaling_factor + bias_val;
        v2 = tanhf(v2) * scaling_factor + bias_val;
        v3 = tanhf(v3) * scaling_factor + bias_val;
        max_val = fmaxf(max_val, fmaxf(fmaxf(v0, v1), fmaxf(v2, v3)));
    }
    
    // Row 3
    {
        const float* row = input_plane + (h_start + 3) * in_width + w_start;
        float v0 = row[0];
        float v1 = row[1];
        float v2 = row[2];
        float v3 = row[3];
        v0 = tanhf(v0) * scaling_factor + bias_val;
        v1 = tanhf(v1) * scaling_factor + bias_val;
        v2 = tanhf(v2) * scaling_factor + bias_val;
        v3 = tanhf(v3) * scaling_factor + bias_val;
        max_val = fmaxf(max_val, fmaxf(fmaxf(v0, v1), fmaxf(v2, v3)));
    }
    
    // Write output
    int out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
    output[out_idx] = max_val;
}

torch::Tensor fused_tanh_scale_bias_maxpool_hip(
    torch::Tensor input,
    torch::Tensor bias,
    float scaling_factor,
    int pool_size
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    
    const int out_height = in_height / pool_size;
    const int out_width = in_width / pool_size;
    
    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());
    
    // 2D block: 16x16 threads
    dim3 block(16, 16);
    dim3 grid(
        (out_width + block.x - 1) / block.x,
        (out_height + block.y - 1) / block.y,
        batch_size * channels
    );
    
    fused_tanh_scale_bias_maxpool_kernel_v5<<<grid, block>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        batch_size,
        channels,
        in_height,
        in_width,
        out_height,
        out_width
    );
    
    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_tanh_scale_bias_maxpool_hip(
    torch::Tensor input,
    torch::Tensor bias,
    float scaling_factor,
    int pool_size
);
"""

fused_module = load_inline(
    name="fused_tanh_scale_bias_maxpool_v5",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_tanh_scale_bias_maxpool_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses tanh, scaling, bias addition, and max-pooling into a single kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.pool_kernel_size = pool_kernel_size
        self.fused_op = fused_module

    def forward(self, x):
        # Convolution (use PyTorch's optimized implementation)
        x = self.conv(x)
        # Fused: tanh + scaling + bias + maxpool
        x = self.fused_op.fused_tanh_scale_bias_maxpool_hip(
            x, 
            self.bias.view(-1),  # Flatten bias to 1D
            self.scaling_factor,
            self.pool_kernel_size
        )
        return x


def get_inputs():
    return [torch.rand(128, 8, 256, 256).cuda()]


def get_init_inputs():
    return [8, 64, 3, 2.0, (64, 1, 1), 4]
