import os
os.environ["CXX"] = "hipcc"
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv2d_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void conv2d_kernel(const float* input, const float* weight, const float* bias, float* output,
  int N, int Cin, int Cout, int H, int W, int K, int Hout, int Wout) {
  const int wout = blockIdx.x * blockDim.x + threadIdx.x;
  const int hout = blockIdx.y * blockDim.y + threadIdx.y;
  const int bco = blockIdx.z;
  const int n = bco / Cout;
  const int cout = bco % Cout;

  if (hout < Hout && wout < Wout && n < N) {
    float sum = bias[cout];
    for (int cin = 0; cin < Cin; ++cin) {
      for (int dy = 0; dy < K; ++dy) {
        int hin = hout + dy;
        for (int dx = 0; dx < K; ++dx) {
          int win = wout + dx;
          sum += input[n * (Cin * H * W) + cin * (H * W) + hin * W + win] *
                 weight[cout * (Cin * K * K) + cin * (K * K) + dy * K + dx];
        }
      }
    }
    output[n * (Cout * Hout * Wout) + cout * (Hout * Wout) + hout * Wout + wout] = sum;
  }
}

torch::Tensor conv2d_hip(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int K) {
  auto N_ = input.size(0);
  auto Cin_ = input.size(1);
  auto H_ = input.size(2);
  auto W_ = input.size(3);
  auto Cout_ = weight.size(0);
  auto Hout_ = H_ - K + 1;
  auto Wout_ = W_ - K + 1;
  int N = static_cast<int>(N_);
  int Cin = static_cast<int>(Cin_);
  int H = static_cast<int>(H_);
  int W = static_cast<int>(W_);
  int Cout = static_cast<int>(Cout_);
  int Hout = static_cast<int>(Hout_);
  int Wout = static_cast<int>(Wout_);
  auto output = torch::empty({N_, Cout_, Hout_, Wout_}, input.options());
  const int TX = 32;
  const int TY = 32;
  dim3 block(TX, TY);
  dim3 grid((Wout + TX - 1) / TX, (Hout + TY - 1) / TY, N * Cout);
  conv2d_kernel<<<grid, block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), N, Cin, Cout, H, W, K, Hout, Wout);
  return output;
}
"""

custom_conv = load_inline(
    name="conv2d",
    cpp_sources=conv2d_cpp_source,
    functions=["conv2d_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor
        self.K = kernel_size
        self.custom_conv = custom_conv

    def forward(self, x):
        weight = self.conv.weight
        bias = self.conv.bias
        conv_out = self.custom_conv.conv2d_hip(x, weight, bias, self.K)
        x = self.bn(conv_out)
        return x * self.scaling_factor
