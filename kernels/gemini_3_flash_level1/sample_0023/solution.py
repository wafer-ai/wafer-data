
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os
import time

os.environ["CXX"] = "hipcc"

# Create a unique name for each compilation to avoid cache issues
unique_name = f"conv2d_module_{int(time.time())}"

conv2d_cpp_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int in_h, int in_w,
    int out_channels, int out_h, int out_w,
    int kernel_size, int stride, int padding)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.y;
    int batch = blockIdx.z;
    
    if (tid < out_h * out_w && oc < out_channels && batch < batch_size) {
        int oh = tid / out_w;
        int ow = tid % out_w;
        
        float sum = (bias != nullptr) ? bias[oc] : 0.0f;
        int ih_start = oh * stride - padding;
        int iw_start = ow * stride - padding;
        
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                int ih = ih_start + kh;
                if (ih >= 0 && ih < in_h) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int iw = iw_start + kw;
                        if (iw >= 0 && iw < in_w) {
                            sum += input[((batch * in_channels + ic) * in_h + ih) * in_w + iw] * 
                                   weight[((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw];
                        }
                    }
                }
            }
        }
        output[((batch * out_channels + oc) * out_h + oh) * out_w + ow] = sum;
    }
}

torch::Tensor conv2d_hip(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding)
{
    input = input.contiguous();
    weight = weight.contiguous();
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);

    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, input.options());

    dim3 block_dim(256);
    dim3 grid_dim((out_h * out_w + 255) / 256, out_channels, batch_size);
    
    float* bias_ptr = (bias.numel() > 0) ? bias.data_ptr<float>() : nullptr;

    conv2d_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size, in_channels, in_h, in_w,
        out_channels, out_h, out_w,
        kernel_size, stride, padding
    );

    return output;
}
"""

conv2d_module = load_inline(
    name=unique_name,
    cpp_sources=conv2d_cpp_source,
    functions=["conv2d_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.stride = 4
        self.padding = 2

    def forward(self, x):
        bias = self.conv1.bias if self.conv1.bias is not None else torch.tensor([], device=x.device, dtype=x.dtype)
        return conv2d_module.conv2d_hip(x, self.conv1.weight, bias, self.stride, self.padding)

def get_inputs():
    batch_size = 256
    return [torch.rand(batch_size, 3, 224, 224).cuda()]

def get_init_inputs():
    return [1000]
