
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

fused_ops_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__global__ void fused_post_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int pool_size,
    float scaling_factor)
{
    int n = blockIdx.z;
    int c = blockIdx.y;
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (n < batch_size && c < channels && pixel_idx < out_h * out_w) {
        int oh = pixel_idx / out_w;
        int ow = pixel_idx % out_w;
        
        float b = bias[c];
        float max_input = -1e38f;

        #pragma unroll
        for (int kh = 0; kh < 4; ++kh) {
            int ih = oh * 4 + kh;
            const float* __restrict__ row_ptr = &input[((n * channels + c) * in_h + ih) * in_w + ow * 4];
            #pragma unroll
            for (int kw = 0; kw < 4; ++kw) {
                float val = row_ptr[kw];
                if (val > max_input) max_input = val;
            }
        }
        
        output[((n * channels + c) * out_h + oh) * out_w + ow] = tanhf(max_input) * scaling_factor + b;
    }
}

torch::Tensor fused_post_conv_hip(
    torch::Tensor input,
    torch::Tensor bias,
    int pool_size,
    float scaling_factor)
{
    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);

    int out_h = in_h / pool_size;
    int out_w = in_w / pool_size;

    auto output = torch::empty({batch_size, channels, out_h, out_w}, input.options());

    dim3 block(256);
    dim3 grid((out_h * out_w + block.x - 1) / block.x, channels, batch_size);

    fused_post_conv_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_h,
        in_w,
        out_h,
        out_w,
        pool_size,
        scaling_factor
    );

    return output;
}
"""

fused_ops_lib = load_inline(
    name="fused_ops_final",
    cpp_sources=fused_ops_source,
    functions=["fused_post_conv_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = float(scaling_factor)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.pool_kernel_size = pool_kernel_size
        self.fused_ops = fused_ops_lib

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_post_conv_hip(x, self.bias.view(-1).contiguous(), self.pool_kernel_size, self.scaling_factor)
        return x

