import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Use im2col + GEMM approach - fixed version
conv2d_cpp_source = """
torch::Tensor conv2d_hip_gemm(torch::Tensor input, torch::Tensor weight, int stride, int padding, int kernel_size);
"""

conv2d_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Optimized im2col kernel
__global__ void im2col_kernel(
    const float* __restrict__ data_im,
    float* __restrict__ data_col,
    const int n, // batch size
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int output_h,
    const int output_w
) {
    // Each thread handles one element in the output col matrix
    const int col_size = channels * kernel_h * kernel_w * output_h * output_w;
    
    for (int batch = 0; batch < n; batch++) {
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < col_size; idx += blockDim.x * gridDim.x) {
            // Decode col index
            const int w_col = idx % output_w;
            int tmp = idx / output_w;
            const int h_col = tmp % output_h;
            tmp = tmp / output_h;
            const int c_col = tmp;
            
            // Map to im
            const int c_im = c_col / (kernel_h * kernel_w);
            const int kh = (c_col / kernel_w) % kernel_h;
            const int kw = c_col % kernel_w;
            
            const int h_im = h_col * stride_h - pad_h + kh;
            const int w_im = w_col * stride_w - pad_w + kw;
            
            float val = 0.0f;
            if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width) {
                val = data_im[batch * channels * height * width + c_im * height * width + h_im * width + w_im];
            }
            
            // col layout: [batch, C*kH*kW, oH*oW]
            data_col[batch * col_size + c_col * (output_h * output_w) + h_col * output_w + w_col] = val;
        }
    }
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
    
    // col shape: [batch_size, in_channels * kernel_h * kernel_w, out_height * out_width]
    auto col = torch::empty({batch_size, in_channels * kernel_h * kernel_w, out_height * out_width}, input.options());
    
    // Launch im2col kernel
    const int col_size = in_channels * kernel_h * kernel_w * out_height * out_width;
    const int block_size = 256;
    const int grid_size = min((col_size + block_size - 1) / block_size, 65535);
    
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
    
    // Use torch.einsum or bmm for batched matrix multiplication
    // weight: [O, K] where K = in_channels * kernel_h * kernel_w
    // col: [B, K, HW] where HW = out_height * out_width
    // output: [B, O, HW]
    
    // Use batched matmul: weight.unsqueeze(0) @ col
    auto output = torch::matmul(weight_reshaped, col);  // Broadcasting: [O, K] @ [B, K, HW] -> [B, O, HW]
    
    // Reshape output to [batch_size, out_channels, out_height, out_width]
    output = output.view({batch_size, out_channels, out_height, out_width});
    
    return output;
}
"""

conv2d_module = load_inline(
    name="conv2d_hip_gemm_v4",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_hip_source,
    functions=["conv2d_hip_gemm"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
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
