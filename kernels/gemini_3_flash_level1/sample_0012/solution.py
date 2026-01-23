
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

conv2d_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int in_h, int in_w, int out_h, int out_w,
    int kernel_size, int stride, int padding, int dilation, int groups,
    bool has_bias) {

    long long total_output_pixels = (long long)batch_size * out_channels * out_h * out_w;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_output_pixels) return;

    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int oc = (idx / ((long long)out_w * out_h)) % out_channels;
    int n = idx / ((long long)out_w * out_h * out_channels);

    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;
    int g = oc / out_channels_per_group;
    int ic_start = g * in_channels_per_group;

    float sum = 0.0f;

    for (int ic_idx = 0; ic_idx < in_channels_per_group; ++ic_idx) {
        int ic = ic_start + ic_idx;
        for (int kh = 0; kh < kernel_size; ++kh) {
            int ih = oh * stride + kh * dilation - padding;
            if (ih >= 0 && ih < in_h) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int iw = ow * stride + kw * dilation - padding;
                    if (iw >= 0 && iw < in_w) {
                        long long input_idx = ((((long long)n * in_channels + ic) * in_h + ih) * in_w) + iw;
                        long long weight_idx = ((((long long)oc * in_channels_per_group + ic_idx) * kernel_size + kh) * kernel_size) + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    if (has_bias) {
        sum += bias[oc];
    }

    output[idx] = sum;
}

torch::Tensor conv2d_hip(
    torch::Tensor input,
    torch::Tensor weight,
    at::Tensor bias,
    int stride, int padding, int dilation, int groups) {

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);

    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int out_h = (in_h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_w = (in_w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, input.options());

    long long total_output_pixels = (long long)batch_size * out_channels * out_h * out_w;
    int block_size = 256;
    long long num_blocks = (total_output_pixels + block_size - 1) / block_size;

    const float* bias_ptr = (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr;
    bool has_bias = (bias_ptr != nullptr);

    hipLaunchKernelGGL(conv2d_kernel, dim3((unsigned int)num_blocks), dim3(block_size), 0, 0,
        input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr, output.data_ptr<float>(),
        batch_size, in_channels, out_channels, in_h, in_w, out_h, out_w,
        kernel_size, stride, padding, dilation, groups, has_bias);

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
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.conv2d.bias if self.conv2d.bias is not None else torch.empty(0, device=x.device)
        return conv2d_module.conv2d_hip(x, self.conv2d.weight, bias, self.stride, self.padding, self.dilation, self.groups)

def get_inputs():
    batch_size = 16
    in_channels = 16
    out_channels = 128
    kernel_size = 3
    width = 1024
    height = 1024
    x = torch.rand(batch_size, in_channels, height, width).cuda()
    return [x]

def get_init_inputs():
    return [16, 128, 3]
