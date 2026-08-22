import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

depthwise_conv_cpp_source = """
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define TILE_SIZE 16
#define KERNEL_SIZE 3
#define BLOCK_SIZE 256

__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int out_height,
    int out_width,
    bool has_bias
) {
    int batch = blockIdx.z;
    int channel = blockIdx.y;
    
    // Calculate output position for this thread
    int total_out_pixels = out_height * out_width;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch >= batch_size || channel >= in_channels || tid >= total_out_pixels) {
        return;
    }
    
    int out_h = tid / out_width;
    int out_w = tid % out_width;
    
    // Calculate input position
    int in_h_start = out_h * stride - padding;
    int in_w_start = out_w * stride - padding;
    
    float sum = 0.0f;
    
    // Convolution with correct weight indexing
    // Weight shape for depthwise conv: (out_channels, in_channels/groups, kernel_h, kernel_w)
    // For depthwise: (in_channels, 1, kernel_h, kernel_w)
    // Weight indexing in row-major (C-contiguous): channel * 1 * kernel_h * kernel_w + kh * kernel_w + kw
    
    int weight_idx_base = channel * kernel_size * kernel_size;
    
    for (int kh = 0; kh < kernel_size; kh++) {
        int in_h = in_h_start + kh;
        if (in_h >= 0 && in_h < height) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int in_w = in_w_start + kw;
                if (in_w >= 0 && in_w < width) {
                    // Input index in row-major
                    // For shape (N, C, H, W): ((n * C + c) * H + h) * W + w
                    int input_idx = ((batch * in_channels + channel) * height + in_h) * width + in_w;
                    // Weight index
                    int weight_idx = weight_idx_base + kh * kernel_size + kw;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias
    if (has_bias) {
        sum += bias[channel];
    }
    
    // Output index: (batch, channel, out_h, out_w)
    int output_idx = ((batch * in_channels + channel) * out_height + out_h) * out_width + out_w;
    output[output_idx] = sum;
}

torch::Tensor depthwise_conv2d_hip(torch::Tensor input, torch::Tensor weight, torch::optional<torch::Tensor> bias, int stride, int padding) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto kernel_size = weight.size(2);
    
    auto out_height = (height + 2 * padding - kernel_size) / stride + 1;
    auto out_width = (width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, in_channels, out_height, out_width}, input.options());
    
    int total_out_pixels = out_height * out_width;
    
    dim3 blockDim(BLOCK_SIZE);
    int total_threads = (total_out_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 gridDim(total_threads, in_channels, batch_size);
    
    bool has_bias = bias.has_value();
    
    depthwise_conv2d_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        out_height,
        out_width,
        has_bias
    );
    
    return output;
}
"""

depthwise_conv = load_inline(
    name="depthwise_conv",
    cpp_sources=depthwise_conv_cpp_source,
    functions=["depthwise_conv2d_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution operation with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        
        # Use Conv2d to initialize weights correctly, then we'll use custom kernel
        # This ensures weights match the reference
        self.conv_ref = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
        
        # Store weights as parameters for potential gradient computation
        self.weight = self.conv_ref.weight
        if bias:
            self.bias_param = self.conv_ref.bias
        else:
            self.register_parameter('bias_param', None)
        
        self.depthwise_conv = depthwise_conv
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using custom HIP kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        # Ensure tensors are contiguous
        x_contiguous = x.contiguous()
        weight_contiguous = self.conv_ref.weight.contiguous()
        
        # Call custom HIP kernel
        bias_tensor = self.conv_ref.bias if self.conv_ref.bias is not None else None
        
        output = self.depthwise_conv.depthwise_conv2d_hip(x_contiguous, weight_contiguous, bias_tensor, self.stride, self.padding)
        
        return output