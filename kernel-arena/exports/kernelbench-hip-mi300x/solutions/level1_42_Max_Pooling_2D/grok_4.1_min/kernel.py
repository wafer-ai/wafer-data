import os
os.environ["CXX"] = "hipcc"

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>

__global__ void maxpool2d_kernel(const float* input, float* output, int N, int C, int H, int W, int Oh, int Ow, int ksize, int stride, int pad, int dil) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_out = N * C * Oh * Ow;
    if (idx >= total_out) return;

    unsigned int stride_nc = C * Oh * Ow;
    int n = idx / stride_nc;
    unsigned int temp = idx % stride_nc;
    unsigned int stride_c = Oh * Ow;
    int c = temp / stride_c;
    unsigned int temp2 = temp % stride_c;
    int oh = temp2 / Ow;
    int ow = temp2 % Ow;

    float max_val = -3.402823466e+38F;
    #pragma unroll 4
    for(int kh = 0; kh < ksize; ++kh) {
        int ih = oh * stride + kh * dil - pad;
        if(ih < 0 || ih >= H) continue;
        #pragma unroll 4
        for(int kw = 0; kw < ksize; ++kw) {
            int iw = ow * stride + kw * dil - pad;
            if(iw < 0 || iw >= W) continue;
            unsigned int in_idx = ((n * C + c) * H + ih) * W + iw;
            max_val = fmaxf(max_val, input[in_idx]);
        }
    }
    unsigned int out_idx = ((n * C + c) * Oh + oh) * Ow + ow;
    output[out_idx] = max_val;
}

torch::Tensor maxpool2d_hip(torch::Tensor input, int ksize, int stride, int pad, int dil) {
    auto sizes = input.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int H = sizes[2];
    int W = sizes[3];

    int k_eff = (ksize - 1) * dil + 1;
    int Oh = (H + 2 * pad - k_eff) / stride + 1;
    int Ow = (W + 2 * pad - k_eff) / stride + 1;

    auto output = torch::empty({N, C, Oh, Ow}, input.options());

    int64_t out_size = (int64_t) N * C * Oh * Ow;
    if (out_size == 0) return output;

    const int block_size = 1024;
    int64_t num_blocks_ll = (out_size + block_size - 1) / block_size;
    dim3 blocks((uint32_t)num_blocks_ll);
    dim3 threads(block_size);

    maxpool2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W, Oh, Ow, ksize, stride, pad, dil);
    return output;
}
"""

def load_maxpool():
    return load_inline(
        name="maxpool2d_v3",
        cpp_sources=cpp_source,
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
        self.maxpool2d = load_maxpool()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool2d.maxpool2d_hip(x, self.kernel_size, self.stride, self.padding, self.dilation)
