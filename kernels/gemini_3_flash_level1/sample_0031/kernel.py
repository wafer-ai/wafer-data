
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Set ROCm compiler
os.environ["CXX"] = "hipcc"

avg_pool2d_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void avg_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long batch_size,
    long long channels,
    long long height,
    long long width,
    long long out_height,
    long long out_width,
    int kernel_size,
    int stride,
    int padding) 
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total_elements = batch_size * channels * out_height * out_width;

    if (idx < total_elements) {
        long long pw = idx % out_width;
        long long ph = (idx / out_width) % out_height;
        long long pc = (idx / (out_width * out_height)) % channels;
        long long pb = idx / (out_width * out_height * channels);

        long long h_start = ph * stride - padding;
        long long w_start = pw * stride - padding;
        long long h_end = h_start + kernel_size;
        long long w_end = w_start + kernel_size;

        float sum = 0.0f;
        
        for (long long h = h_start; h < h_end; ++h) {
            for (long long w = w_start; w < w_end; ++w) {
                if (h >= 0 && h < height && w >= 0 && w < width) {
                    sum += input[((pb * channels + pc) * height + h) * width + w];
                }
            }
        }
        output[idx] = sum / (kernel_size * kernel_size);
    }
}

torch::Tensor avg_pool2d_hip(torch::Tensor input, int kernel_size, int stride, int padding) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);

    int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());

    long long total_elements = (long long)batch_size * channels * out_height * out_width;
    const int block_size = 256;
    const long long num_blocks = (total_elements + block_size - 1) / block_size;

    hipLaunchKernelGGL(avg_pool2d_kernel, dim3(num_blocks), dim3(block_size), 0, 0,
                       input.data_ptr<float>(),
                       output.data_ptr<float>(),
                       batch_size, channels, height, width,
                       out_height, out_width,
                       kernel_size, stride, padding);

    return output;
}
"""

avg_pool2d_module = load_inline(
    name="avg_pool2d",
    cpp_sources=avg_pool2d_source,
    functions=["avg_pool2d_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input tensor might not be contiguous, better make it contiguous
        if not x.is_contiguous():
            x = x.contiguous()
        return avg_pool2d_module.avg_pool2d_hip(x, self.kernel_size, self.stride, self.padding)

batch_size = 16
channels = 64
height = 2048
width = 2048
kernel_size = 11

def get_inputs():
    x = torch.rand(batch_size, channels, height, width).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size]
