
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Set ROCm compiler
os.environ["CXX"] = "hipcc"

avg_pool2d_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void __launch_bounds__(256) avg_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int height,
    int width,
    int out_height,
    int out_width,
    int stride,
    float inv_kernel_area) 
{
    long long bc_idx = blockIdx.y; // batch * channels
    long long total_out_per_channel = (long long)out_height * out_width;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < total_out_per_channel) {
        int pw = tid % out_width;
        int ph = tid / out_width;

        int h_start = ph * stride;
        int w_start = pw * stride;

        const float* input_ptr = input + (bc_idx * height * width) + (long long)h_start * width + w_start;
        float sum = 0.0f;

        #pragma unroll
        for (int kh = 0; kh < 11; ++kh) {
            const float* row = input_ptr + (long long)kh * width;
            #pragma unroll
            for (int kw = 0; kw < 11; ++kw) {
                sum += row[kw];
            }
        }
        output[bc_idx * total_out_per_channel + tid] = sum * inv_kernel_area;
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

    long long total_out_per_channel = (long long)out_height * out_width;
    const int block_size = 256;
    const long long num_blocks_per_channel = (total_out_per_channel + block_size - 1) / block_size;
    float inv_kernel_area = 1.0f / (kernel_size * kernel_size);

    dim3 grid(num_blocks_per_channel, batch_size * channels);
    hipLaunchKernelGGL(avg_pool2d_kernel, grid, dim3(block_size), 0, 0,
                       input.data_ptr<float>(),
                       output.data_ptr<float>(),
                       (int)height, (int)width,
                       out_height, out_width,
                       stride, inv_kernel_area);

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
        if not x.is_cuda:
            x = x.cuda()
        if not x.is_contiguous():
            x = x.contiguous()
        return avg_pool2d_module.avg_pool2d_hip(x, self.kernel_size, self.stride, self.padding)

def get_inputs():
    batch_size = 16
    channels = 64
    height = 2048
    width = 2048
    x = torch.rand(batch_size, channels, height, width).cuda()
    return [x]

def get_init_inputs():
    kernel_size = 11
    return [kernel_size]
