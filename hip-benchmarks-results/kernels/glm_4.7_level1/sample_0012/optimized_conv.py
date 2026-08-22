import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

conv2d_cpp_source = """
#include <hip/hip_runtime.h>

#define BLOCK_SIZE 16

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height_in,
    const int width_in,
    const int height_out,
    const int width_out,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation) {
    
    // 2D thread block for output tiles
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Calculate global output position
    int h_out = blockIdx.y * BLOCK_SIZE + ty;
    int w_out = blockIdx.x * BLOCK_SIZE + tx;
    int oc = blockIdx.z;
    
    if (h_out >= height_out || w_out >= width_out) return;
    
    // Calculate input patch start position
    int h_in_start = h_out * stride - padding;
    int w_in_start = w_out * stride - padding;
    
    // Each thread computes one output element
    float sum = 0.0f;
    
    // Unroll inner loops for better performance
    for (int ic = 0; ic < in_channels; ++ic) {
        int in_index_base = ic * height_in * width_in;
        int weight_index_base = oc * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size;
        
        // Manual unrolling of 3x3 kernel
        for (int kh = 0; kh < kernel_size; ++kh) {
            int h_in = h_in_start + kh * dilation;
            if (h_in >= 0 && h_in < height_in) {
                int in_row_index = in_index_base + h_in * width_in;
                
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int w_in = w_in_start + kw * dilation;
                    if (w_in >= 0 && w_in < width_in) {
                        int in_idx = in_row_index + w_in;
                        int weight_idx = weight_index_base + kh * kernel_size + kw;
                        sum += input[in_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Write output
    int out_idx = blockIdx.z * height_out * width_out + h_out * width_out + w_out;
    output[out_idx] = sum;
}

torch::Tensor conv2d_hip(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height_in = input.size(2);
    const int width_in = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int height_out = (height_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int width_out = (width_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, height_out, width_out}, input.options());
    
    // Use 2D thread blocks
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((width_out + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                (height_out + BLOCK_SIZE - 1) / BLOCK_SIZE,
                out_channels);
    
    // Process each batch separately for simplicity
    for (int b = 0; b < batch_size; ++b) {
        const float* input_ptr = input.data_ptr<float>() + b * in_channels * height_in * width_in;
        float* output_ptr = output.data_ptr<float>() + b * out_channels * height_out * width_out;
        
        hipLaunchKernelGGL(
            conv2d_kernel,
            blocks,
            threads,
            0,
            0,
            input_ptr,
            weight.data_ptr<float>(),
            output_ptr,
            1,  // batch_size for this call
            in_channels,
            out_channels,
            height_in,
            width_in,
            height_out,
            width_out,
            kernel_size,
            stride,
            padding,
            dilation
        );
    }
    
    return output;
}
"""

conv2d_module = load_inline(
    name="conv2d_hip",
    cpp_sources=conv2d_cpp_source,
    functions=["conv2d_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with a square input and square kernel using optimized HIP kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias_flag = bias
        
        # Initialize weights (same as nn.Conv2d default initialization)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=(5 ** 0.5))
        
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
            fan_in = in_channels * kernel_size * kernel_size
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias_param, -bound, bound)
        else:
            self.register_parameter('bias_param', None)
        
        self.conv2d_hip = conv2d_module
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using optimized HIP kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Ensure tensors are on the same device
        if not x.is_cuda:
            x = x.cuda()
        
        output = self.conv2d_hip.conv2d_hip(x, self.weight, self.stride, self.padding, self.dilation)
        
        # Add bias if present
        if self.bias_param is not None:
            if not self.bias_param.is_cuda:
                self.bias_param = self.bias_param.cuda()
            output += self.bias_param.view(1, -1, 1, 1)
        
        return output

# Test code
batch_size = 16
in_channels = 16
out_channels = 128
kernel_size = 3
width = 1024
height = 1024

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]