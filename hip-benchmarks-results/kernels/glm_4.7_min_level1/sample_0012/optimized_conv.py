import os
import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

conv2d_hip_source = """
#include <hip/hip_runtime.h>

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int H_in, const int W_in, const int H_out, const int W_out,
    const int K_h, const int K_w,
    const int stride, const int pad, const int dilation) {
    
    const int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int c_out = blockIdx.z;
    
    if (w_out >= W_out || h_out >= H_out) return;
    
    int h_in_start = h_out * stride - pad;
    int w_in_start = w_out * stride - pad;
    
    for (int n = 0; n < N; ++n) {
        float sum = 0.0f;
        
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int kh = 0; kh < K_h; ++kh) {
                int h_in = h_in_start + kh * dilation;
                if (h_in < 0 || h_in >= H_in) continue;
                
                for (int kw = 0; kw < K_w; ++kw) {
                    int w_in = w_in_start + kw * dilation;
                    if (w_in < 0 || w_in >= W_in) continue;
                    
                    int idx_in = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                    int idx_w = ((c_out * C_in + c_in) * K_h + kh) * K_w + kw;
                    
                    sum += input[idx_in] * weight[idx_w];
                }
            }
        }
        
        int idx_out = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
        output[idx_out] = sum + (bias ? bias[c_out] : 0.0f);
    }
}

torch::Tensor conv2d_hip(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups) {
    
    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int C_out = weight.size(0);
    int K_h = weight.size(2);
    int K_w = weight.size(3);
    
    int H_out = (H_in + 2 * pad_h - dilation_h * (K_h - 1) - 1) / stride_h + 1;
    int W_out = (W_in + 2 * pad_w - dilation_w * (K_w - 1) - 1) / stride_w + 1;
    
    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());
    
    dim3 blockDim(16, 16);
    dim3 gridDim((W_out + 15) / 16, (H_out + 15) / 16, C_out);
    
    const float* bias_ptr = bias.numel() > 0 ? bias.data_ptr<float>() : nullptr;
    
    hipLaunchKernelGGL(conv2d_kernel, gridDim, blockDim, 0, 0,
        input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr,
        output.data_ptr<float>(),
        N, C_in, C_out, H_in, W_in, H_out, W_out,
        K_h, K_w, stride_h, pad_h, dilation_h);
    
    return output;
}
"""

conv2d_hip = load_inline(
    name="conv2d_hip",
    cpp_sources=conv2d_hip_source,
    functions=["conv2d_hip"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.conv2d_hip_impl = conv2d_hip
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        bias_tensor = self.bias if self.bias is not None else torch.empty(0, device=x.device)
        return self.conv2d_hip_impl.conv2d_hip(x, self.weight, bias_tensor,
            self.stride, self.stride, self.padding, self.padding,
            self.dilation, self.dilation, self.groups)