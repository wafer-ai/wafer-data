
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

conv2d_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void conv2d_optimized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int in_h, int in_w,
    int out_channels, int out_h, int out_w,
    int k_h, int k_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dil_h, int dil_w,
    bool has_bias) {

    int total_output_pixels = batch_size * out_h * out_w;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.y;

    if (index < total_output_pixels) {
        int ow = index % out_w;
        int oh = (index / out_w) % out_h;
        int b = index / (out_w * out_h);

        float val = has_bias ? bias[oc] : 0.0f;
        
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < k_h; ++kh) {
                for (int kw = 0; kw < k_w; ++kw) {
                    int ih = oh * stride_h - pad_h + kh * dil_h;
                    int iw = ow * stride_w - pad_w + kw * dil_w;

                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                        int input_idx = ((b * in_channels + ic) * in_h + ih) * in_w + iw;
                        int weight_idx = ((oc * in_channels + ic) * k_h + kh) * k_w + kw;
                        val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        output[((b * out_channels + oc) * out_h + oh) * out_w + ow] = val;
    }
}

torch::Tensor conv2d_hip_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dil_h, int dil_w) {

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);

    int out_channels = weight.size(0);
    int k_h = weight.size(2);
    int k_w = weight.size(3);

    int out_h = (in_h + 2 * pad_h - dil_h * (k_h - 1) - 1) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - dil_w * (k_w - 1) - 1) / stride_w + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, input.options());

    int pixels_per_batch = out_h * out_w;
    int total_output_pixels = batch_size * pixels_per_batch;
    
    dim3 block_size(256);
    dim3 num_blocks((total_output_pixels + 255) / 256, out_channels);

    float* bias_ptr = (bias.has_value()) ? bias.value().data_ptr<float>() : nullptr;
    bool has_bias = bias.has_value();

    conv2d_optimized_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size, in_channels, in_h, in_w,
        out_channels, out_h, out_w,
        k_h, k_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        has_bias);

    return output;
}
"""

conv2d_lib = load_inline(
    name="conv2d_lib_final",
    cpp_sources=conv2d_hip_source,
    functions=["conv2d_hip_forward"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv2d_lib.conv2d_hip_forward(
            x.contiguous(), self.conv2d.weight.contiguous(), 
            self.conv2d.bias.contiguous() if self.conv2d.bias is not None else None,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1]
        )

