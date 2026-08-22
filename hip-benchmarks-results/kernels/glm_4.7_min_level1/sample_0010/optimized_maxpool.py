import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

maxpool2d_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void maxpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int out_height,
    const int out_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    const int out_size = out_height * out_width;
    const int channel = (idx / out_size) % channels;
    const int spatial_idx = idx % out_size;
    const int out_y = spatial_idx / out_width;
    const int out_x = spatial_idx % out_width;
    const int batch = idx / (out_size * channels);
    
    if (out_x >= out_width || out_y >= out_height || channel >= channels || batch >= batch_size) {
        return;
    }
    
    // Compute input coordinates
    const int y_start = out_y * stride - padding;
    const int x_start = out_x * stride - padding;
    
    float max_val = -1e20f;
    
    // Unrolled 4x4 max pool (kernel_size=4)
    #pragma unroll
    for (int ky = 0; ky < 4; ++ky) {
        const int y = y_start + ky * dilation;
        const bool y_valid = (y >= 0 && y < height);
        
        #pragma unroll
        for (int kx = 0; kx < 4; ++kx) {
            if (y_valid) {
                const int x = x_start + kx * dilation;
                if (x >= 0 && x < width) {
                    const int input_idx = ((batch * channels + channel) * height + y) * width + x;
                    max_val = fmaxf(max_val, input[input_idx]);
                }
            }
        }
    }
    
    const int output_idx = ((batch * channels + channel) * out_height + out_y) * out_width + out_x;
    output[output_idx] = max_val;
}

torch::Tensor maxpool2d_hip(
    torch::Tensor input,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    const int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, channels, out_height, out_width}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size * channels * out_height * out_width + block_size - 1) / block_size;
    
    maxpool2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        dilation
    );
    
    return output;
}
"""

maxpool2d = load_inline(
    name="maxpool2d",
    cpp_sources=maxpool2d_cpp_source,
    functions=["maxpool2d_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Simple model that performs Max Pooling 2D.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the Max Pooling 2D layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.maxpool2d = maxpool2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 2D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D.
        """
        return self.maxpool2d.maxpool2d_hip(x, self.kernel_size, self.stride, self.padding, self.dilation)