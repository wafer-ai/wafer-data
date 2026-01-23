import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused kernel with better thread organization for AMD GPUs
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define POOL_SIZE 4

// Optimized kernel with explicit unrolling for pool_size=4
__global__ __launch_bounds__(512) void fused_tanh_scale_bias_maxpool_kernel_v4(
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
    // Each thread computes one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_height * out_width;
    
    if (idx >= total) return;
    
    // Decode output indices
    int ow = idx % out_width;
    int oh = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int b = idx / (out_width * out_height * channels);
    
    float bias_val = __ldg(&bias[c]);
    
    int h_start = oh * POOL_SIZE;
    int w_start = ow * POOL_SIZE;
    
    // Base index for this batch and channel
    int plane_stride = in_height * in_width;
    int base_idx = (b * channels + c) * plane_stride + h_start * in_width + w_start;
    
    // Compute max over 4x4 pooling window
    float max_val = -INFINITY;
    
    // Row 0
    {
        float v0 = __ldg(&input[base_idx]);
        float v1 = __ldg(&input[base_idx + 1]);
        float v2 = __ldg(&input[base_idx + 2]);
        float v3 = __ldg(&input[base_idx + 3]);
        v0 = tanhf(v0) * scaling_factor + bias_val;
        v1 = tanhf(v1) * scaling_factor + bias_val;
        v2 = tanhf(v2) * scaling_factor + bias_val;
        v3 = tanhf(v3) * scaling_factor + bias_val;
        max_val = fmaxf(max_val, fmaxf(fmaxf(v0, v1), fmaxf(v2, v3)));
    }
    
    // Row 1
    {
        int row_idx = base_idx + in_width;
        float v0 = __ldg(&input[row_idx]);
        float v1 = __ldg(&input[row_idx + 1]);
        float v2 = __ldg(&input[row_idx + 2]);
        float v3 = __ldg(&input[row_idx + 3]);
        v0 = tanhf(v0) * scaling_factor + bias_val;
        v1 = tanhf(v1) * scaling_factor + bias_val;
        v2 = tanhf(v2) * scaling_factor + bias_val;
        v3 = tanhf(v3) * scaling_factor + bias_val;
        max_val = fmaxf(max_val, fmaxf(fmaxf(v0, v1), fmaxf(v2, v3)));
    }
    
    // Row 2
    {
        int row_idx = base_idx + 2 * in_width;
        float v0 = __ldg(&input[row_idx]);
        float v1 = __ldg(&input[row_idx + 1]);
        float v2 = __ldg(&input[row_idx + 2]);
        float v3 = __ldg(&input[row_idx + 3]);
        v0 = tanhf(v0) * scaling_factor + bias_val;
        v1 = tanhf(v1) * scaling_factor + bias_val;
        v2 = tanhf(v2) * scaling_factor + bias_val;
        v3 = tanhf(v3) * scaling_factor + bias_val;
        max_val = fmaxf(max_val, fmaxf(fmaxf(v0, v1), fmaxf(v2, v3)));
    }
    
    // Row 3
    {
        int row_idx = base_idx + 3 * in_width;
        float v0 = __ldg(&input[row_idx]);
        float v1 = __ldg(&input[row_idx + 1]);
        float v2 = __ldg(&input[row_idx + 2]);
        float v3 = __ldg(&input[row_idx + 3]);
        v0 = tanhf(v0) * scaling_factor + bias_val;
        v1 = tanhf(v1) * scaling_factor + bias_val;
        v2 = tanhf(v2) * scaling_factor + bias_val;
        v3 = tanhf(v3) * scaling_factor + bias_val;
        max_val = fmaxf(max_val, fmaxf(fmaxf(v0, v1), fmaxf(v2, v3)));
    }
    
    output[idx] = max_val;
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
    
    const int total = batch_size * channels * out_height * out_width;
    const int block_size = 512;
    const int num_blocks = (total + block_size - 1) / block_size;
    
    fused_tanh_scale_bias_maxpool_kernel_v4<<<num_blocks, block_size>>>(
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
    name="fused_tanh_scale_bias_maxpool_v4",
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
