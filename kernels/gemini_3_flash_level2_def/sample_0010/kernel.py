
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

# HIP kernel for fused Tanh, Scaling, Bias addition, and Max-pooling
fused_ops_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

__global__ void fused_pool_kernel(
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
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_h * out_w;

    if (tid < total_elements) {
        int ow = tid % out_w;
        int oh = (tid / out_w) % out_h;
        int c = (tid / (out_w * out_h)) % channels;
        int n = tid / (out_w * out_h * channels);

        int start_h = oh * pool_size;
        int start_w = ow * pool_size;

        float max_val = -1e38f;

        const float* input_ptr = input + (n * channels + c) * in_h * in_w;

        for (int ph = 0; ph < pool_size; ++ph) {
            for (int pw = 0; pw < pool_size; ++pw) {
                int ih = start_h + ph;
                int iw = start_w + pw;
                // Since pool_size * out_h <= in_h, ih < in_h and iw < in_w
                // should always be true if pool_size is a factor of in_h/in_w
                // but let's be safe.
                if (ih < in_h && iw < in_w) {
                    float val = input_ptr[ih * in_w + iw];
                    if (val > max_val) {
                        max_val = val;
                    }
                }
            }
        }

        float res = tanhf(max_val) * scaling_factor + bias[c];
        output[tid] = res;
    }
}

torch::Tensor fused_pool_hip(
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

    int total_elements = batch_size * channels * out_h * out_w;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    hipLaunchKernelGGL(fused_pool_kernel, dim3(num_blocks), dim3(block_size), 0, 0,
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

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_kernel_source,
    functions=["fused_pool_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size).cuda()
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape).cuda())
        self.pool_kernel_size = pool_kernel_size
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_pool_hip(x, self.bias.view(-1), self.pool_kernel_size, self.scaling_factor)
        return x

def get_inputs():
    batch_size = 128
    in_channels = 8
    height, width = 256, 256
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    in_channels = 8
    out_channels = 64
    kernel_size = 3
    scaling_factor = 2.0
    bias_shape = (out_channels, 1, 1)
    pool_kernel_size = 4
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size]
