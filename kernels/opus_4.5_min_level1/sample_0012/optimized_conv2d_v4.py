import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Use explicit im2col + GEMM with rocBLAS - fixed version
conv2d_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Im2col kernel - produce column matrix for one batch
__global__ void im2col_kernel(
    const float* __restrict__ data_im,
    float* __restrict__ data_col,
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int height_col,
    const int width_col,
    const int total_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    const int output_spatial = height_col * width_col;
    const int w_out = idx % width_col;
    const int h_out = (idx / width_col) % height_col;
    const int c_in_kernel = idx / output_spatial;  // which channel*kh*kw element
    
    const int c_in = c_in_kernel / (kernel_h * kernel_w);
    const int k_idx = c_in_kernel % (kernel_h * kernel_w);
    const int kh = k_idx / kernel_w;
    const int kw = k_idx % kernel_w;
    
    const int h_in = h_out * stride_h - pad_h + kh * dilation_h;
    const int w_in = w_out * stride_w - pad_w + kw * dilation_w;
    
    float val = 0.0f;
    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
        val = data_im[(c_in * height + h_in) * width + w_in];
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
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_spatial = out_height * out_width;
    
    // Reshape weight to 2D: [out_channels, in_channels * kernel_size * kernel_size]
    const int K = in_channels * kernel_size * kernel_size;
    auto weight_2d = weight.view({out_channels, K});
    
    // Allocate im2col buffer for single batch element
    const int col_size = K * out_spatial;
    auto col = torch::empty({K, out_spatial}, input.options());
    
    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (col_size + block_size - 1) / block_size;
    
    for (int b = 0; b < batch_size; ++b) {
        // im2col for this batch element
        im2col_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>() + b * in_channels * in_height * in_width,
            col.data_ptr<float>(),
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
            out_width,
            col_size
        );
        
        // Wait for kernel to complete before GEMM
        hipDeviceSynchronize();
        
        // GEMM: output = weight_2d @ col
        // weight_2d: [out_channels, K]
        // col: [K, out_spatial]
        // result: [out_channels, out_spatial]
        auto out_2d = torch::mm(weight_2d, col);
        
        // Copy to output
        output[b] = out_2d.view({out_channels, out_height, out_width});
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
    name="conv2d_hip_v4",
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
