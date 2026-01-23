import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Use explicit im2col + GEMM with rocBLAS
conv2d_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <rocblas/rocblas.h>

// Im2col kernel for extracting image patches
__global__ void im2col_kernel(
    const float* __restrict__ data_im,
    float* __restrict__ data_col,
    int batch,
    int channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int height_col,
    int width_col
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = channels * kernel_h * kernel_w * height_col * width_col;
    
    if (idx >= total) return;
    
    // Calculate position
    int w_out = idx % width_col;
    int h_out = (idx / width_col) % height_col;
    int channel_in_offset = idx / (width_col * height_col);
    int channel_in = channel_in_offset / (kernel_h * kernel_w);
    int kh = (channel_in_offset / kernel_w) % kernel_h;
    int kw = channel_in_offset % kernel_w;
    
    int h_in = h_out * stride_h - pad_h + kh * dilation_h;
    int w_in = w_out * stride_w - pad_w + kw * dilation_w;
    
    float val = 0.0f;
    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
        val = data_im[((batch * channels + channel_in) * height + h_in) * width + w_in];
    }
    
    data_col[idx] = val;
}

torch::Tensor conv2d_hip(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Reshape weight to 2D: [out_channels, in_channels * kernel_size * kernel_size]
    auto weight_2d = weight.reshape({out_channels, in_channels * kernel_size * kernel_size});
    
    // Allocate im2col buffer
    int col_size = in_channels * kernel_size * kernel_size * out_height * out_width;
    auto col = torch::empty({col_size}, input.options());
    
    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (col_size + block_size - 1) / block_size;
    
    for (int b = 0; b < batch_size; ++b) {
        // im2col for this batch element
        im2col_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>() + b * in_channels * in_height * in_width,
            col.data_ptr<float>(),
            b,
            in_channels,
            in_height,
            in_width,
            kernel_size,
            kernel_size,
            padding,
            padding,
            stride,
            stride,
            dilation,
            dilation,
            out_height,
            out_width
        );
        
        // GEMM: output = weight_2d @ col
        // output shape: [out_channels, out_height * out_width]
        // weight_2d shape: [out_channels, in_channels * kernel_size^2]
        // col shape: [in_channels * kernel_size^2, out_height * out_width]
        auto col_2d = col.reshape({in_channels * kernel_size * kernel_size, out_height * out_width});
        auto out_2d = torch::mm(weight_2d, col_2d);
        
        // Copy to output
        output[b] = out_2d.reshape({out_channels, out_height, out_width});
    }
    
    return output;
}
"""

conv2d_cpp_source = """
torch::Tensor conv2d_hip(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation
);
"""

conv2d_module = load_inline(
    name="conv2d_hip_v3",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_hip_source,
    functions=["conv2d_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights same as nn.Conv2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = conv2d_module.conv2d_hip(
            x.contiguous(),
            self.weight.contiguous(),
            self.stride,
            self.padding,
            self.dilation
        )
        
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        
        return output
