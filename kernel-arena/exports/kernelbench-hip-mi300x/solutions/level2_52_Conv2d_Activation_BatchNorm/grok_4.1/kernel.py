import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

batch_size = 64
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3

custom_activation_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__device__ float softplus(float input, float beta, float threshold) {
  float z = beta * input - threshold;
  if (z > threshold) {
    return input;
  } else {
    float exp_z = (z < 0.0f) ? expf(z) : 1.0f;
    float sp;
    if (z >= 0.0f) {
      sp = (z + log1pf(exp_z)) / beta;
    } else {
      sp = log1pf(exp_z) / beta;
    }
    return sp;
  }
}

__global__ void custom_activation_kernel(const float* input, float* output, size_t size) {
  size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
  if (idx >= size) return;
  float x = input[idx];
  float sp = softplus(x, 1.0f, 20.0f);
  float tx = tanhf(sp);
  output[idx] = x * tx;
}

torch::Tensor custom_activation_hip(torch::Tensor input) {
  torch::Tensor output = torch::empty_like(input);
  size_t size = static_cast<size_t>(input.numel());
  const int block_size = 256;
  const size_t max_blocks_per_launch = 65535;
  size_t offset = 0;
  float* in_ptr = input.data_ptr<float>();
  float* out_ptr = output.data_ptr<float>();
  while (offset < size) {
    size_t remaining = size - offset;
    size_t cur_nblocks = (remaining + block_size - 1) / block_size;
    if (cur_nblocks > max_blocks_per_launch) cur_nblocks = max_blocks_per_launch;
    dim3 block(block_size);
    dim3 grid(cur_nblocks);
    size_t cur_size = cur_nblocks * block_size;
    if (cur_size > remaining) cur_size = remaining;
    custom_activation_kernel<<<grid, block>>>(in_ptr + offset, out_ptr + offset, cur_size);
    offset += cur_size;
  }
  hipDeviceSynchronize();
  return output;
}
"""

custom_act = load_inline(
    name="custom_act",
    cpp_sources=custom_activation_cpp_source,
    functions=["custom_activation_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.custom_act = custom_act

    def forward(self, x):
        x = self.conv(x)
        x = self.custom_act.custom_activation_hip(x)
        x = self.bn(x)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
