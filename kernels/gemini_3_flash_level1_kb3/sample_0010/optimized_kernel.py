
import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

maxpool2d_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <float.h>

__global__ void maxpool2d_kernel_v4(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int height,
    int width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    int total_output_elements = batch_size * channels * output_height * output_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_output_elements) {
        int ow = idx % output_width;
        int oh = (idx / output_width) % output_height;
        int c_n = idx / (output_width * output_height);

        int h_start = oh * stride - padding;
        int w_start = ow * stride - padding;

        float max_val = -FLT_MAX;
        const float* input_ptr = input + (c_n * height * width);

        for (int i = 0; i < kernel_size; ++i) {
            int h = h_start + i * dilation;
            if (h >= 0 && h < height) {
                const float* row_ptr = input_ptr + h * width;
                for (int j = 0; j < kernel_size; ++j) {
                    int w = w_start + j * dilation;
                    if (w >= 0 && w < width) {
                        float val = row_ptr[w];
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }
            }
        }
        output[idx] = max_val;
    }
}

torch::Tensor maxpool2d_hip(torch::Tensor input, int kernel_size, int stride, int padding, int dilation) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    int output_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());

    int total_output_elements = batch_size * channels * output_height * output_width;
    const int block_size = 512;
    const int num_blocks = (total_output_elements + block_size - 1) / block_size;

    hipLaunchKernelGGL(maxpool2d_kernel_v4, dim3(num_blocks), dim3(block_size), 0, 0,
                        input.data_ptr<float>(), output.data_ptr<float>(),
                        batch_size, channels, height, width, output_height, output_width,
                        kernel_size, stride, padding, dilation);

    return output;
}
"""

maxpool2d_cpp_header = """
torch::Tensor maxpool2d_hip(torch::Tensor input, int kernel_size, int stride, int padding, int dilation);
"""

maxpool2d_lib = load_inline(
    name="maxpool2d_lib_v4",
    cpp_sources=maxpool2d_cpp_header,
    cuda_sources=maxpool2d_cpp_source,
    functions=["maxpool2d_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return maxpool2d_lib.maxpool2d_hip(x, self.kernel_size, self.stride, self.padding, self.dilation)

def get_inputs():
    batch_size = 32
    channels = 64
    height = 512
    width = 512
    x = torch.rand(batch_size, channels, height, width).cuda()
    return [x]

def get_init_inputs():
    kernel_size = 4
    stride = 1
    padding = 1
    dilation = 1
    return [kernel_size, stride, padding, dilation]
