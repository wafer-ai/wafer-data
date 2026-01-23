import os
os.environ['CXX'] = 'hipcc'
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

activation_cpp = '''
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void custom_activation_kernel(const float* input, float* output, size_t size) {
    size_t idx = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
    if (idx < size) {
        float x = input[idx];
        float sp;
        if (x > 0.0f) {
            sp = x + log1pf(expf(-x));
        } else {
            sp = log1pf(expf(x));
        }
        float t = tanhf(sp);
        output[idx] = t * x;
    }
}

torch::Tensor custom_activation_hip(torch::Tensor input) {
    auto output = torch::empty_like(input);
    size_t size = input.numel();
    const int threads = 1024;
    dim3 block(threads);
    dim3 grid((size + threads - 1) / threads);
    custom_activation_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}
'''

activation_ext = load_inline(
    name='activation_ext',
    cpp_sources=activation_cpp,
    functions=['custom_activation_hip'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.activation = activation_ext

    def forward(self, x):
        x = self.conv(x)
        x = self.activation.custom_activation_hip(x)
        x = self.bn(x)
        return x

batch_size = 64
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
