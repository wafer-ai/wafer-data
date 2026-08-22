
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

depthwise_conv2d_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void depthwise_conv2d_kernel_optimized_v2(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int K, int stride, int padding,
    int H_out, int W_out) {

    int nc = blockIdx.x; 
    int h_out = blockIdx.y;
    int w_out_start = blockIdx.z * 256;
    int w_out = w_out_start + threadIdx.x;

    int c = nc % C;
    int n = nc / C;

    if (w_out < W_out) {
        float sum = 0.0f;
        int h_start = h_out * stride - padding;
        int w_start = w_out * stride - padding;

        const float* weight_ptr = &weight[c * K * K];
        const float* input_nc_ptr = &input[(n * C + c) * H * W];

        #pragma unroll
        for (int kh = 0; kh < K; ++kh) {
            int h_in = h_start + kh;
            if (h_in >= 0 && h_in < H) {
                const float* input_ptr = &input_nc_ptr[h_in * W];
                #pragma unroll
                for (int kw = 0; kw < K; ++kw) {
                    int w_in = w_start + kw;
                    if (w_in >= 0 && w_in < W) {
                        sum += input_ptr[w_in] * weight_ptr[kh * K + kw];
                    }
                }
            }
        }
        if (bias) {
            sum += bias[c];
        }
        output[(nc * H_out + h_out) * W_out + w_out] = sum;
    }
}

torch::Tensor depthwise_conv2d_hip(
    torch::Tensor input,
    torch::Tensor weight,
    at::optional<torch::Tensor> bias,
    int stride,
    int padding) {

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int K = weight.size(2);

    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;

    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    dim3 grid(N * C, H_out, (W_out + 255) / 256);
    dim3 block(256);

    float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;

    hipLaunchKernelGGL(depthwise_conv2d_kernel_optimized_v2, grid, block, 0, 0,
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C, H, W,
        K, stride, padding,
        H_out, W_out);

    return output;
}
"""

depthwise_conv2d_module = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    functions=["depthwise_conv2d_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.conv2d.bias if self.conv2d.bias is not None else None
        return depthwise_conv2d_module.depthwise_conv2d_hip(x, self.conv2d.weight, bias, self.stride, self.padding)
