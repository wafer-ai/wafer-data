import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

maxpool_cpp_source = """
#include <hip/hip_runtime.h>

const float NEG_INF = -3.402823466e+38f;

__global__ void maxpool2d_kernel(
    const float* input,
    float* output,
    const int N,
    const int C,
    const int Hin,
    const int Win,
    const int Hout,
    const int Wout,
    const int kh,
    const int kw,
    const int sh,
    const int sw,
    const int ph,
    const int pw,
    const int dh,
    const int dw
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * Hout * Wout) return;

    int houtwout = Hout * Wout;
    int temp = idx / houtwout;
    int n = temp / C;
    int c = temp % C;
    int temp2 = idx % houtwout;
    int oh = temp2 / Wout;
    int ow = temp2 % Wout;

    float max_val = NEG_INF;

#pragma unroll
    for (int y = 0; y < kh; y++) {
#pragma unroll
        for (int x = 0; x < kw; x++) {
            int ih = oh * sh - ph + y * dh;
            int iw = ow * sw - pw + x * dw;
            if (ih >= 0 && ih < Hin && iw >= 0 && iw < Win) {
                int in_idx = ((n * C + c) * Hin + ih) * Win + iw;
                max_val = fmaxf(max_val, input[in_idx]);
            }
        }
    }

    int out_idx = ((n * C + c) * Hout + oh) * Wout + ow;
    output[out_idx] = max_val;
}

torch::Tensor maxpool2d_hip(torch::Tensor input, int kernel_size, int stride, int padding, int dilation) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto Hin = input.size(2);
    auto Win = input.size(3);

    int kh = kernel_size;
    int kw = kernel_size;
    int sh = stride;
    int sw = stride;
    int ph = padding;
    int pw = padding;
    int dh = dilation;
    int dw = dilation;

    int Hout = (Hin + 2 * ph - (dh * (kh - 1) + 1)) / sh + 1;
    int Wout = (Win + 2 * pw - (dw * (kw - 1) + 1)) / sw + 1;

    if (Hout < 0) Hout = 0;
    if (Wout < 0) Wout = 0;

    auto output = torch::empty({N, C, Hout, Wout}, input.options());

    size_t numel = 1LL * N * C * (size_t)Hout * Wout;
    const int block_size = 512;
    dim3 block(block_size);
    dim3 grid((numel + block_size - 1) / block_size);

    maxpool2d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, Hin, Win, Hout, Wout,
        kh, kw, sh, sw, ph, pw, dh, dw
    );

    return output;
}
"""

maxpool_module = load_inline(
    name="maxpool2d",
    cpp_sources=maxpool_cpp_source,
    functions=["maxpool2d_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.maxpool_hip = maxpool_module.maxpool2d_hip
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_hip(x, self.kernel_size, self.stride, self.padding, self.dilation)

def get_inputs():
    batch_size = 32
    channels = 64
    height = 512
    width = 512
    x = torch.rand(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    kernel_size = 4
    stride = 1
    padding = 1
    dilation = 1
    return [kernel_size, stride, padding, dilation]
