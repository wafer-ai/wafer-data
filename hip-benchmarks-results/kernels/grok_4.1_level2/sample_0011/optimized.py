import os
import torch
import torch.nn as nn
import torch.nn.init as nninit
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

cpp_source = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void conv2d_kernel(const float *input, const float *weight, const float *bias, float *output, int B, int IC, int OC, int Hin, int Win, int Hout, int Wout, int KH, int KW) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total_out = static_cast<size_t>(B) * OC * Hout * Wout;
    if (idx >= total_out) return;
    int posout_size = Hout * Wout;
    int bc = static_cast<int>(idx / posout_size);
    int b = bc / OC;
    int oc = bc % OC;
    int houtw = static_cast<int>(idx % posout_size);
    int hout = houtw / Wout;
    int wout = houtw % Wout;
    float accum = bias[oc];
    for (int i = 0; i < IC; i++) {
        for (int ky = 0; ky < KH; ky++) {
            int hi = hout + ky;
            if (hi >= Hin) continue;
            for (int kx = 0; kx < KW; kx++) {
                int wi = wout + kx;
                if (wi >= Win) continue;
                size_t idxin = static_cast<size_t>((b * IC + i) * Hin + hi) * Win + wi;
                size_t widx = static_cast<size_t>((oc * IC + i) * KH + ky) * KW + kx;
                accum += weight[widx] * input[idxin];
            }
        }
    }
    output[idx] = accum;
}

torch::Tensor custom_conv2d_hip(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
  auto i_sizes = input.sizes();
  int64_t B = i_sizes[0];
  int64_t IC = i_sizes[1];
  int64_t Hin = i_sizes[2];
  int64_t Win = i_sizes[3];
  auto w_sizes = weight.sizes();
  int64_t OC = w_sizes[0];
  int64_t KH = w_sizes[2];
  int64_t KW = w_sizes[3];
  int64_t Hout = Hin - KH + 1;
  int64_t Wout = Win - KW + 1;
  auto out = torch::zeros({B, OC, Hout, Wout}, input.options());
  int64_t total_out = B * OC * Hout * Wout;
  const int block_size = 256;
  dim3 block(block_size);
  dim3 grid(static_cast<unsigned int>((total_out + block_size - 1LL) / block_size));
  hipLaunchKernelGGL(conv2d_kernel, grid, block, 0, 0, input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), static_cast<int>(B), static_cast<int>(IC), static_cast<int>(OC), static_cast<int>(Hin), static_cast<int>(Win), static_cast<int>(Hout), static_cast<int>(Wout), static_cast<int>(KH), static_cast<int>(KW));
  return out;
}
"""

custom_ops = load_inline(
    name="custom_conv2d",
    cpp_sources=cpp_source,
    functions=["custom_conv2d_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv_weight = nn.Parameter(torch.empty((out_channels, in_channels, kernel_size, kernel_size)))
        nninit.kaiming_normal_(self.conv_weight, mode='fan_out', nonlinearity='relu')
        self.conv_bias = nn.Parameter(torch.empty(out_channels))
        nninit.zeros_(self.conv_bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.custom_ops = custom_ops

    def forward(self, x):
        x = self.custom_ops.custom_conv2d_hip(x, self.conv_weight, self.conv_bias)
        x = self.group_norm(x)
        x = x * self.scale
        x = self.maxpool(x)
        x = torch.clamp(x, self.clamp_min, self.clamp_max)
        return x
