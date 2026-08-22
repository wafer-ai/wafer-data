import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Use im2col + GEMM approach with rocBLAS
conv2d_cpp_source = """
torch::Tensor conv2d_hip_gemm(torch::Tensor input, torch::Tensor weight, int stride, int padding, int kernel_size);
"""

conv2d_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <hipblas/hipblas.h>

// im2col kernel - extracts patches from input
__global__ void im2col_kernel(
    const float* __restrict__ input,
    float* __restrict__ col,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int out_h,
    const int out_w
) {
    const int total = batch_size * channels * kernel_h * kernel_w * out_h * out_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total) return;
    
    // Decode the index
    const int w_out = idx % out_w;
    int tmp = idx / out_w;
    const int h_out = tmp % out_h;
    tmp = tmp / out_h;
    const int k_col = tmp % (kernel_h * kernel_w);
    tmp = tmp / (kernel_h * kernel_w);
    const int c = tmp % channels;
    const int n = tmp / channels;
    
    const int kh = k_col / kernel_w;
    const int kw = k_col % kernel_w;
    
    const int h_in = h_out * stride_h - pad_h + kh;
    const int w_in = w_out * stride_w - pad_w + kw;
    
    float val = 0.0f;
    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
        val = input[n * channels * height * width + c * height * width + h_in * width + w_in];
    }
    
    // col layout: [batch, channels*kernel_h*kernel_w, out_h*out_w]
    col[n * (channels * kernel_h * kernel_w * out_h * out_w) + 
        (c * kernel_h * kernel_w + k_col) * (out_h * out_w) + 
        h_out * out_w + w_out] = val;
}

torch::Tensor conv2d_hip_gemm(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int kernel_size
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    
    input = input.contiguous();
    weight = weight.contiguous();
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    
    const int out_channels = weight.size(0);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    
    const int out_height = (in_height + 2 * padding - kernel_h) / stride + 1;
    const int out_width = (in_width + 2 * padding - kernel_w) / stride + 1;
    
    // Create im2col matrix for all batches
    // col shape: [batch_size, in_channels * kernel_h * kernel_w, out_height * out_width]
    auto col = torch::empty({batch_size, in_channels * kernel_h * kernel_w, out_height * out_width}, input.options());
    
    // Launch im2col kernel
    const int total_elements = batch_size * in_channels * kernel_h * kernel_w * out_height * out_width;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    
    im2col_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        col.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        kernel_h,
        kernel_w,
        padding,
        padding,
        stride,
        stride,
        out_height,
        out_width
    );
    
    // Reshape weight to [out_channels, in_channels * kernel_h * kernel_w]
    auto weight_reshaped = weight.view({out_channels, in_channels * kernel_h * kernel_w});
    
    // Perform batched matrix multiplication
    // weight: [out_channels, in_channels * kernel_h * kernel_w]
    // col: [batch_size, in_channels * kernel_h * kernel_w, out_height * out_width]
    // output: [batch_size, out_channels, out_height * out_width]
    auto output = torch::matmul(weight_reshaped, col);
    
    // Reshape output to [batch_size, out_channels, out_height, out_width]
    output = output.view({batch_size, out_channels, out_height, out_width});
    
    return output;
}
"""

conv2d_module = load_inline(
    name="conv2d_hip_gemm",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_hip_source,
    functions=["conv2d_hip_gemm"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
    extra_ldflags=["-lhipblas"]
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dilation == 1 and self.groups == 1 and self.bias is None:
            return conv2d_module.conv2d_hip_gemm(x, self.weight, self.stride, self.padding, self.kernel_size)
        else:
            return torch.nn.functional.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def get_inputs():
    x = torch.rand(16, 16, 1024, 1024).cuda()
    return [x]


def get_init_inputs():
    return [16, 128, 3]
