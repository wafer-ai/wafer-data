import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel: tanh + scaling + bias addition
tanh_scale_bias_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void tanh_scale_bias_kernel(
    const float* input,
    const float* bias,
    float* output,
    int num_elements,
    int channels,
    int hw_size,
    float scaling_factor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements) {
        // Calculate channel index from linear index
        // Maps: idx -> (batch, channel, height, width)
        int channel_idx = (idx % (channels * hw_size)) / hw_size;
        
        // Apply fused operation: tanh + scaling + bias
        float val = input[idx];
        val = tanhf(val);
        val *= scaling_factor;
        val += bias[channel_idx];
        output[idx] = val;
    }
}

extern "C" void tanh_scale_bias_hip(
    torch::Tensor input,
    torch::Tensor bias,
    torch::Tensor output,
    float scaling_factor
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int num_elements = batch_size * channels * height * width;
    int hw_size = height * width;

    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    tanh_scale_bias_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements,
        channels,
        hw_size,
        scaling_factor
    );
}
"""

tanh_scale_bias = load_inline(
    name="tanh_scale_bias",
    cpp_sources=tanh_scale_bias_cpp_source,
    functions=["tanh_scale_bias_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model with fused tanh+scaling+bias kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size) -> None:
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)
        self.tanh_scale_bias = tanh_scale_bias

    def forward(self, x):
        # Convolution (keep PyTorch's optimized implementation)
        x = self.conv(x)
        # Fused element-wise operations: tanh + scaling + bias
        output = torch.empty_like(x)
        self.tanh_scale_bias.tanh_scale_bias_hip(x, self.bias, output, self.scaling_factor)
        # Max-pooling (keep PyTorch's optimized implementation)
        x = self.max_pool(output)
        return x