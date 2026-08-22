import os
os.environ["CXX"] = "hipcc"
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.cpp_extension import load_inline

depthwise_conv2d_cpp = """
#include <hip/hip_runtime.h>

__global__ void depthwise_conv2d_kernel(const float *input, int B, int C, int H, int W, const float *weight, int K, float *output, int Ho, int Wo, int S, int P) {
  extern __shared__ float shmem[];
  float* input_tile = shmem;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bc = blockIdx.z;
  int b = bc / C;
  int c = bc % C;
  int ch_stride = H * W;
  int out_ch_stride = Ho * Wo;
  int ih_base = (blockIdx.x * blockDim.x) * S - P;
  int iw_base = (blockIdx.y * blockDim.y) * S - P;
  int tile_h = blockDim.x + K - 1;
  int tile_w = blockDim.y + K - 1;
  int in_ch_offset = b * C * ch_stride + c * ch_stride;
  int out_ch_offset = b * C * out_ch_stride + c * out_ch_stride;
  // Load input tile into shared memory
  for (int r = ty; r < tile_h; r += blockDim.y) {
    int ih = ih_base + r;
    bool valid_ih = (ih >= 0 && ih < H);
    for (int c_ = tx; c_ < tile_w; c_ += blockDim.x) {
      int iw = iw_base + c_;
      float val = 0.0f;
      if (valid_ih && iw >= 0 && iw < W) {
        val = input[in_ch_offset + ih * W + iw];
      }
      input_tile[r * tile_w + c_] = val;
    }
  }
  __syncthreads();
  // Compute
  int oh = blockIdx.x * blockDim.x + tx;
  int ow = blockDim.y * blockIdx.y + ty;
  if (oh < Ho && ow < Wo) {
    float sum = 0.0f;
    for (int kh = 0; kh < K; ++kh) {
      int ih = S * oh + kh - P;
      if (ih < 0 || ih >= H) continue;
      int r = ih - ih_base;
      for (int kw = 0; kw < K; ++kw) {
        int iw = S * ow + kw - P;
        if (iw < 0 || iw >= W) continue;
        int cc = iw - iw_base;
        float inp = input_tile[r * tile_w + cc];
        int widx = c * K * K + kh * K + kw;
        sum += inp * weight[widx];
      }
    }
    output[out_ch_offset + oh * Wo + ow] = sum;
  }
}

torch::Tensor depthwise_conv2d_hip(torch::Tensor input, torch::Tensor weight, int stride, int pad) {
  int B = input.size(0);
  int C = input.size(1);
  int H = input.size(2);
  int W = input.size(3);
  int K = weight.size(2);
  int Ho = (H + 2 * pad - K) / stride + 1;
  int Wo = (W + 2 * pad - K) / stride + 1;
  auto out = torch::empty({B, C, Ho, Wo}, input.options());
  const int TX = 16;
  const int TY = 16;
  dim3 threads(TX, TY);
  size_t shmem_size = 32 * 32 * sizeof(float); // safe
  dim3 blocks((Ho + TX - 1) / TX, (Wo + TY - 1) / TY, B * C);
  depthwise_conv2d_kernel<<<blocks, threads, shmem_size>>>(input.data_ptr<float>(), B, C, H, W, weight.data_ptr<float>(), K, out.data_ptr<float>(), Ho, Wo, stride, pad);
  return out;
}
"""

depthwise_conv_module = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp,
    functions=["depthwise_conv2d_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        k = kernel_size
        self.weight = nn.Parameter(torch.empty(in_channels, 1, k, k))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.hip_conv = depthwise_conv_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hip_conv.depthwise_conv2d_hip(x, self.weight, self.stride, self.padding)

batch_size = 16
in_channels = 64
kernel_size = 3
width = 512
height = 512
stride = 1
padding = 0

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, kernel_size, stride, padding]
