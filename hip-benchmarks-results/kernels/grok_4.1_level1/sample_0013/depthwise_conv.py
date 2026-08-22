import os
os.environ["CXX"] = "hipcc"
import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline
import torch.nn.init as init

hip_source = """
#include <hip/hip_runtime.h>

__global__ void depthwise_conv2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int B, int C, int H, int W, int K, int stride, int padding, int outH, int outW
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)B * C * outH * outW;
    if ((uint64_t)tid >= total) return;
    
    unsigned int temp = tid;
    int ow = temp % outW;
    temp /= outW;
    int oh = temp % outH;
    temp /= outH;
    int c = temp % C;
    temp /= C;
    int b = temp % B;
    
    float acc = 0.0f;
    int ih0 = oh * stride - padding;
    int iw0 = ow * stride - padding;
    for (int kh = 0; kh < K; kh++) {
        int ih = ih0 + kh;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < K; kw++) {
            int iw = iw0 + kw;
            if (iw < 0 || iw >= W) continue;
            int input_offset = ((b * C + c) * H + ih) * W + iw;
            int weight_offset = c * K * K + kh * K + kw;
            acc += input[input_offset] * weight[weight_offset];
        }
    }
    int output_offset = ((b * C + c) * outH + oh) * outW + ow;
    output[output_offset] = acc;
}

torch::Tensor depthwise_conv2d_hip(torch::Tensor input, torch::Tensor weight, int stride, int padding) {
    auto sizes = input.sizes();
    int64_t B = sizes[0];
    int64_t C = sizes[1];
    int64_t H = sizes[2];
    int64_t W = sizes[3];
    int64_t K = weight.size(2);
    int64_t outH = (H + 2LL * padding - K) / stride + 1;
    int64_t outW = (W + 2LL * padding - K) / stride + 1;
    torch::Tensor output = torch::empty({B, C, outH, outW}, input.options());
    
    const int TPB = 256;
    int64_t total = B * C * outH * outW;
    uint32_t num_blocks = (total + TPB - 1) / TPB;
    
    dim3 grid(num_blocks);
    dim3 block(TPB);
    
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(depthwise_conv2d_kernel),
        grid, block, 0, 0,
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(K),
        stride,
        padding,
        static_cast<int>(outH),
        static_cast<int>(outW)
    );
    return output;
}
"""

depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=hip_source,
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
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, kernel_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return depthwise_conv2d.depthwise_conv2d_hip(x, self.weight, self.stride, self.padding)

def get_inputs():
    batch_size = 16
    in_channels = 64
    height = 512
    width = 512
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    in_channels = 64
    kernel_size = 3
    stride = 1
    padding = 0
    return [in_channels, kernel_size, stride, padding]
