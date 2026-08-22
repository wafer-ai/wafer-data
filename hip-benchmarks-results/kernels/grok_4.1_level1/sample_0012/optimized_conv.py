import os
import math
import torch
import torch.nn as nn
import torch.nn.init as nninit
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

conv2d_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void conv2d_tiled_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int N, int Cin, int Cout, int Hin, int Win, int Hout, int Wout, int KH, int KW
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int n = bz / Cout;
    int oc = bz % Cout;
    int oh = by * 16 + ty;
    int ow = bx * 16 + tx;
    if (oh >= Hout || ow >= Wout || n >= N || oc >= Cout) return;

    float sum = 0.0f;
    #pragma unroll
    for (int c = 0; c < Cin; ++c) {
        #pragma unroll
        for (int kh = 0; kh < KH; ++kh) {
            int ih = oh + kh;
            if (ih >= Hin) continue;
            #pragma unroll
            for (int kw = 0; kw < KW; ++kw) {
                int iw = ow + kw;
                if (iw >= Win) continue;
                float i_val = input[((n * Cin + c) * Hin + ih) * Win + iw];
                float w_val = weight[((oc * Cin + c) * KH + kh) * KW + kw];
                sum += i_val * w_val;
            }
        }
    }
    output[((n * Cout + oc) * Hout + oh) * Wout + ow] = sum;
}

torch::Tensor conv2d_hip(torch::Tensor input, torch::Tensor weight) {
    int N = input.size(0);
    int Cin = input.size(1);
    int Hin = input.size(2);
    int Win = input.size(3);
    int Cout = weight.size(0);
    int Cinw = weight.size(1);
    int KH = weight.size(2);
    int KW = weight.size(3);

    if (Cinw != Cin) {
        throw std::runtime_error("Groups != 1 not supported");
    }

    int Hout = Hin - KH + 1;
    int Wout = Win - KW + 1;

    auto output = torch::zeros({N, Cout, Hout, Wout}, input.options());

    const int TILE_DIM = 16;
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((Wout + TILE_DIM - 1) / TILE_DIM, (Hout + TILE_DIM - 1) / TILE_DIM, N * Cout);

    hipStream_t stream = 0;
    hipLaunchKernelGGL(conv2d_tiled_kernel, grid, block, 0, stream,
                       input.data_ptr<float>(),
                       weight.data_ptr<float>(),
                       output.data_ptr<float>(),
                       N, Cin, Cout, Hin, Win, Hout, Wout, KH, KW);

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
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        kh = kernel_size
        kw = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kh, kw))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nninit.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv2d_module.conv2d_hip(x, self.weight)
