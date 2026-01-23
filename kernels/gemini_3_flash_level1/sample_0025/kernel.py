
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

depthwise_conv2d_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out, int H_in, int W_in,
    int H_out, int W_out,
    int K, int stride, int padding,
    bool has_bias
) {
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_oc = blockIdx.z;
    int n = batch_oc / C_out;
    int oc = batch_oc % C_out;
    int ic = oc / (C_out / C_in);

    if (n < N && oc < C_out && h_out < H_out && w_out < W_out) {
        float val = 0.0f;
        int h_base = h_out * stride - padding;
        int w_base = w_out * stride - padding;

        for (int kh = 0; kh < K; ++kh) {
            int h_in = h_base + kh;
            if (h_in >= 0 && h_in < H_in) {
                for (int kw = 0; kw < K; ++kw) {
                    int w_in = w_base + kw;
                    if (w_in >= 0 && w_in < W_in) {
                        val += input[((static_cast<long long>(n) * C_in + ic) * H_in + h_in) * W_in + w_in] *
                               weight[(static_cast<long long>(oc) * K + kh) * K + kw];
                    }
                }
            }
        }
        if (has_bias) {
            val += bias[oc];
        }
        output[((static_cast<long long>(n) * C_out + oc) * H_out + h_out) * W_out + w_out] = val;
    }
}

torch::Tensor depthwise_conv2d_hip(
    torch::Tensor input,
    torch::Tensor weight,
    at::optional<torch::Tensor> bias,
    int stride,
    int padding
) {
    auto input_c = input.contiguous();
    auto weight_c = weight.contiguous();
    
    int N = input_c.size(0);
    int C_in = input_c.size(1);
    int H_in = input_c.size(2);
    int W_in = input_c.size(3);

    int C_out = weight_c.size(0);
    int K = weight_c.size(2);

    int H_out = (H_in + 2 * padding - K) / stride + 1;
    int W_out = (W_in + 2 * padding - K) / stride + 1;

    auto output = torch::empty({N, C_out, H_out, W_out}, input_c.options());

    dim3 block_dim(16, 16);
    dim3 grid_dim((W_out + block_dim.x - 1) / block_dim.x,
                   (H_out + block_dim.y - 1) / block_dim.y,
                   N * C_out);

    bool has_bias = bias.has_value() && bias.value().defined();
    const float* bias_ptr = nullptr;
    torch::Tensor bias_c;
    if (has_bias) {
        bias_c = bias.value().contiguous();
        bias_ptr = bias_c.data_ptr<float>();
    }

    hipLaunchKernelGGL(depthwise_conv2d_kernel, grid_dim, block_dim, 0, 0,
        input_c.data_ptr<float>(),
        weight_c.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C_in, C_out, H_in, W_in, H_out, W_out,
        K, stride, padding, has_bias
    );

    return output;
}
"""

depthwise_conv2d_module = load_inline(
    name="depthwise_conv2d_final",
    cpp_sources=depthwise_conv2d_cpp_source,
    functions=["depthwise_conv2d_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), stride=stride, padding=padding, groups=in_channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return depthwise_conv2d_module.depthwise_conv2d_hip(
            x, self.conv2d.weight, self.conv2d.bias, self.conv2d.stride[0], self.conv2d.padding[0]
        )
