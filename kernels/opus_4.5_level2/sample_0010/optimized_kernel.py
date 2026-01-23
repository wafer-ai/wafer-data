import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel: tanh + scaling + bias + maxpool
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Fused kernel: applies tanh, scaling, bias addition, and max pooling in one pass
__global__ void fused_tanh_scale_bias_maxpool_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int pool_size,
    const float scaling_factor
) {
    // Each thread computes one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    // Compute output coordinates
    int ow = idx % out_width;
    int oh = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int b = idx / (out_width * out_height * channels);
    
    // Get bias value for this channel
    float bias_val = bias[c];
    
    // Compute max over pooling window
    float max_val = -INFINITY;
    
    int in_row_start = oh * pool_size;
    int in_col_start = ow * pool_size;
    
    for (int ph = 0; ph < pool_size; ph++) {
        for (int pw = 0; pw < pool_size; pw++) {
            int in_row = in_row_start + ph;
            int in_col = in_col_start + pw;
            
            if (in_row < in_height && in_col < in_width) {
                int in_idx = b * (channels * in_height * in_width) + 
                             c * (in_height * in_width) + 
                             in_row * in_width + in_col;
                
                float val = input[in_idx];
                // Apply tanh
                val = tanhf(val);
                // Apply scaling
                val = val * scaling_factor;
                // Apply bias
                val = val + bias_val;
                
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }
    
    output[idx] = max_val;
}

torch::Tensor fused_tanh_scale_bias_maxpool_hip(
    torch::Tensor input,
    torch::Tensor bias,
    int pool_size,
    float scaling_factor
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    
    const int out_height = in_height / pool_size;
    const int out_width = in_width / pool_size;
    
    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());
    
    const int total_elements = batch_size * channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_tanh_scale_bias_maxpool_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_height,
        in_width,
        out_height,
        out_width,
        pool_size,
        scaling_factor
    );
    
    return output;
}
"""

fused_kernel_cpp = """
torch::Tensor fused_tanh_scale_bias_maxpool_hip(
    torch::Tensor input,
    torch::Tensor bias,
    int pool_size,
    float scaling_factor
);
"""

fused_module = load_inline(
    name="fused_tanh_scale_bias_maxpool",
    cpp_sources=fused_kernel_cpp,
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
        self.fused_module = fused_module

    def forward(self, x):
        # Convolution (use PyTorch's optimized implementation)
        x = self.conv(x)
        # Fused: tanh + scaling + bias + maxpool
        x = self.fused_module.fused_tanh_scale_bias_maxpool_hip(
            x, 
            self.bias.view(-1),  # Flatten bias to 1D
            self.pool_kernel_size,
            self.scaling_factor
        )
        return x


def get_inputs():
    return [torch.rand(128, 8, 256, 256).cuda()]


def get_init_inputs():
    return [8, 64, 3, 2.0, (64, 1, 1), 4]
