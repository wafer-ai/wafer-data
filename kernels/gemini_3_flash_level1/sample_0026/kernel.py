
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

conv_transpose2d_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>

__global__ void conv_transpose2d_kernel_stride1(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_h,
    int input_w,
    int kernel_size,
    int padding,
    int groups,
    int output_h,
    int output_w) {

    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_out_idx = blockIdx.z;
    int n = batch_out_idx / out_channels;
    int o = batch_out_idx % out_channels;

    if (n < batch_size && o < out_channels && oh < output_h && ow < output_w) {
        float sum = 0.0f;
        int out_channels_per_group = out_channels / groups;
        int g = o / out_channels_per_group;
        int o_in_group = o % out_channels_per_group;
        int in_channels_per_group = in_channels / groups;

        int kh_min = max(0, oh + padding - (input_h - 1));
        int kh_max = min(kernel_size - 1, oh + padding);
        int kw_min = max(0, ow + padding - (input_w - 1));
        int kw_max = min(kernel_size - 1, ow + padding);

        for (int i_in_group = 0; i_in_group < in_channels_per_group; ++i_in_group) {
            int i = g * in_channels_per_group + i_in_group;
            for (int kh = kh_min; kh <= kh_max; ++kh) {
                int ih = oh + padding - kh;
                for (int kw = kw_min; kw <= kw_max; ++kw) {
                    int iw = ow + padding - kw;
                    int weight_idx = (((i * out_channels_per_group) + o_in_group) * kernel_size + kh) * kernel_size + kw;
                    int input_idx = ((n * in_channels + i) * input_h + ih) * input_w + iw;
                    sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
                }
            }
        }
        output[((n * out_channels + o) * output_h + oh) * output_w + ow] = sum;
    }
}

__global__ void conv_transpose2d_kernel_general(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_h,
    int input_w,
    int kernel_size,
    int stride,
    int padding,
    int groups,
    int output_h,
    int output_w) {

    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_out_idx = blockIdx.z;
    int n = batch_out_idx / out_channels;
    int o = batch_out_idx % out_channels;

    if (n < batch_size && o < out_channels && oh < output_h && ow < output_w) {
        float sum = 0.0f;
        int out_channels_per_group = out_channels / groups;
        int g = o / out_channels_per_group;
        int o_in_group = o % out_channels_per_group;
        int in_channels_per_group = in_channels / groups;

        for (int i_in_group = 0; i_in_group < in_channels_per_group; ++i_in_group) {
            int i = g * in_channels_per_group + i_in_group;
            for (int kh = 0; kh < kernel_size; ++kh) {
                int ih_stride = oh + padding - kh;
                if (ih_stride >= 0 && ih_stride % stride == 0) {
                    int ih = ih_stride / stride;
                    if (ih < input_h) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int iw_stride = ow + padding - kw;
                            if (iw_stride >= 0 && iw_stride % stride == 0) {
                                int iw = iw_stride / stride;
                                if (iw < input_w) {
                                    int weight_idx = (((i * out_channels_per_group) + o_in_group) * kernel_size + kh) * kernel_size + kw;
                                    int input_idx = ((n * in_channels + i) * input_h + ih) * input_w + iw;
                                    sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
                                }
                            }
                        }
                    }
                }
            }
        }
        output[((n * out_channels + o) * output_h + oh) * output_w + ow] = sum;
    }
}

torch::Tensor conv_transpose2d_hip(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int output_padding,
    int groups) {

    input = input.contiguous();
    weight = weight.contiguous();

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_h = input.size(2);
    int input_w = input.size(3);

    int out_channels_per_group = weight.size(1);
    int out_channels = out_channels_per_group * groups;
    int kernel_size = weight.size(2);

    int output_h = (input_h - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_w = (input_w - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({batch_size, out_channels, output_h, output_w}, input.options());

    dim3 block_dim(16, 16);
    dim3 grid_dim((output_w + block_dim.x - 1) / block_dim.x,
                  (output_h + block_dim.y - 1) / block_dim.y,
                  batch_size * out_channels);

    if (stride == 1) {
        hipLaunchKernelGGL(conv_transpose2d_kernel_stride1, grid_dim, block_dim, 0, 0,
                           input.data_ptr<float>(),
                           weight.data_ptr<float>(),
                           output.data_ptr<float>(),
                           batch_size, in_channels, out_channels,
                           input_h, input_w, kernel_size,
                           padding, groups,
                           output_h, output_w);
    } else {
        hipLaunchKernelGGL(conv_transpose2d_kernel_general, grid_dim, block_dim, 0, 0,
                           input.data_ptr<float>(),
                           weight.data_ptr<float>(),
                           output.data_ptr<float>(),
                           batch_size, in_channels, out_channels,
                           input_h, input_w, kernel_size,
                           stride, padding, groups,
                           output_h, output_w);
    }

    if (bias.has_value()) {
        output.add_(bias.value().contiguous().view({1, -1, 1, 1}));
    }

    return output;
}
"""

conv_transpose2d_optimized = load_inline(
    name="conv_transpose2d_optimized",
    cpp_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            return conv_transpose2d_optimized.conv_transpose2d_hip(
                x, self.conv_transpose2d.weight, self.conv_transpose2d.bias,
                self.stride, self.padding, self.output_padding, self.groups
            )
        return self.conv_transpose2d(x)

