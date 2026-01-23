import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

maxpool2d_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <float.h>

__global__ void maxpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    // Calculate output position
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = batch_size * channels * out_height * out_width;
    
    if (idx >= total_out) return;
    
    // Decompose index
    int ow = idx % out_width;
    int oh = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int b = idx / (out_width * out_height * channels);
    
    // Calculate input start position
    int ih_start = oh * stride - padding;
    int iw_start = ow * stride - padding;
    
    float max_val = -FLT_MAX;
    
    // Input pointer for this batch and channel
    const float* input_ptr = input + (b * channels + c) * in_height * in_width;
    
    // Iterate over pooling window
    #pragma unroll
    for (int kh = 0; kh < kernel_size; ++kh) {
        int ih = ih_start + kh * dilation;
        if (ih >= 0 && ih < in_height) {
            #pragma unroll
            for (int kw = 0; kw < kernel_size; ++kw) {
                int iw = iw_start + kw * dilation;
                if (iw >= 0 && iw < in_width) {
                    float val = input_ptr[ih * in_width + iw];
                    max_val = fmaxf(max_val, val);
                }
            }
        }
    }
    
    output[idx] = max_val;
}

torch::Tensor maxpool2d_hip(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    // Calculate output dimensions
    int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());
    
    int total_out = batch_size * channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_out + block_size - 1) / block_size;
    
    maxpool2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_height,
        in_width,
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

maxpool2d_cpp_source = """
torch::Tensor maxpool2d_hip(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
);
"""

maxpool2d_module = load_inline(
    name="maxpool2d_hip",
    cpp_sources=maxpool2d_cpp_source,
    cuda_sources=maxpool2d_hip_source,
    functions=["maxpool2d_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.maxpool2d = maxpool2d_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool2d.maxpool2d_hip(x, self.kernel_size, self.stride, self.padding, self.dilation)


def get_inputs():
    x = torch.rand(32, 64, 512, 512).cuda()
    return [x]


def get_init_inputs():
    return [4, 1, 1, 1]
