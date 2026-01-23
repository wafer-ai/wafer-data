
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

depthwise_conv2d_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

template <int KH, int KW, int UNROLL_H>
__global__ void depthwise_conv2d_kernel_optimized(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int HO, int WO,
    int stride, int padding,
    bool has_bias) {

    __shared__ float s_weight[KH * KW];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int wo = blockIdx.x * blockDim.x + tx;
    int ho_start = (blockIdx.y * blockDim.y + ty) * UNROLL_H;
    int batch_channel_idx = blockIdx.z;
    int c = batch_channel_idx % C;
    int n = batch_channel_idx / C;

    if (ty == 0 && tx < KH * KW) {
        s_weight[tx] = weight[c * KH * KW + tx];
    }
    __syncthreads();

    if (wo < WO) {
        float sums[UNROLL_H];
        #pragma unroll
        for (int i = 0; i < UNROLL_H; ++i) {
            sums[i] = 0.0f;
        }

        const float* input_ptr = input + (n * C + c) * H * W;

        #pragma unroll
        for (int kh = 0; kh < KH; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < KW; ++kw) {
                float w_val = s_weight[kh * KW + kw];
                #pragma unroll
                for (int i = 0; i < UNROLL_H; ++i) {
                    int ho = ho_start + i;
                    if (ho < HO) {
                        int h_in = ho * stride - padding + kh;
                        int w_in = wo * stride - padding + kw;
                        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                            sums[i] += input_ptr[h_in * W + w_in] * w_val;
                        }
                    }
                }
            }
        }

        float b_val = has_bias ? bias[c] : 0.0f;
        #pragma unroll
        for (int i = 0; i < UNROLL_H; ++i) {
            int ho = ho_start + i;
            if (ho < HO) {
                output[((n * C + c) * HO + ho) * WO + wo] = sums[i] + b_val;
            }
        }
    }
}

__global__ void depthwise_conv2d_kernel_generic(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int KH, int KW,
    int HO, int WO,
    int stride, int padding,
    bool has_bias) {

    int wo = blockIdx.x * blockDim.x + threadIdx.x;
    int ho = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_channel_idx = blockIdx.z;
    int c = batch_channel_idx % C;
    int n = batch_channel_idx / C;

    if (wo < WO && ho < HO) {
        float sum = 0.0f;
        const float* input_ptr = input + (n * C + c) * H * W;
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                int h_in = ho * stride - padding + kh;
                int w_in = wo * stride - padding + kw;
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    sum += input_ptr[h_in * W + w_in] * weight[(c * KH + kh) * KW + kw];
                }
            }
        }
        if (has_bias) sum += bias[c];
        output[((n * C + c) * HO + ho) * WO + wo] = sum;
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
    int KH = weight.size(2);
    int KW = weight.size(3);
    
    int HO = (H + 2 * padding - KH) / stride + 1;
    int WO = (W + 2 * padding - KW) / stride + 1;
    
    auto output = torch::empty({N, C, HO, WO}, input.options());
    
    bool has_bias = bias.has_value();
    const float* bias_ptr = has_bias ? bias.value().data_ptr<float>() : nullptr;

    if (KH == 3 && KW == 3) {
        const int UNROLL_H = 4;
        dim3 block_size(32, 8);
        dim3 grid_size((WO + 31) / 32, (HO + 8 * UNROLL_H - 1) / (8 * UNROLL_H), N * C);
        hipLaunchKernelGGL((depthwise_conv2d_kernel_optimized<3, 3, UNROLL_H>), grid_size, block_size, 0, 0,
            input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr, output.data_ptr<float>(),
            N, C, H, W, HO, WO, stride, padding, has_bias);
    } else {
        dim3 block_size(16, 16);
        dim3 grid_size((WO + 15) / 16, (HO + 15) / 16, N * C);
        hipLaunchKernelGGL(depthwise_conv2d_kernel_generic, grid_size, block_size, 0, 0,
            input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr, output.data_ptr<float>(),
            N, C, H, W, KH, KW, HO, WO, stride, padding, has_bias);
    }
    
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
        bias = self.conv2d.bias
        return depthwise_conv2d_module.depthwise_conv2d_hip(x.contiguous(), self.conv2d.weight, bias, self.stride, self.padding)
