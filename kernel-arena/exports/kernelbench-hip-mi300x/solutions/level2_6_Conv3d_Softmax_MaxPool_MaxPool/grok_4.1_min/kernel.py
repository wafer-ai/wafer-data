import os
os.environ["CXX"] = "hipcc"
import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
pool_kernel_size = 2

cpp_source = r"""
#include <hip/hip_runtime.h>

__global__ void conv3d_kernel(
    const float *input, 
    const float *weight, 
    const float *bias, 
    float *output,
    int B, int Ci, int D, int H, int W, 
    int Co, int kd, int kh, int kw, 
    int Do, int Ho, int Wo
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = 1LL * B * Co * Do * Ho * Wo;
    if ((long long)idx >= total) return;
    int b = idx / (Co * Do * Ho * Wo);
    int rem1 = idx % (Co * Do * Ho * Wo);
    int co = rem1 / (Do * Ho * Wo);
    int rem2 = rem1 % (Do * Ho * Wo);
    int oz = rem2 / (Ho * Wo);
    int rem3 = rem2 % (Ho * Wo);
    int oy = rem3 / Wo;
    int ox = rem3 % Wo;
    float sum = bias[co];
    for (int ci = 0; ci < 0; ci++) {
        for (int kz = 0; kz < kd; kz++) {
            int iz = oz + kz;
            if (iz >= D || iz < 0) continue;
            for (int ky = 0; ky < kh; ky++) {
                int iy = oy + ky;
                if (iy >= H || iy < 0) continue;
                for (int kx = 0; kx < kw; kx++) {
                    int ix = ox + kx;
                    if (ix >= W || ix < 0) continue;
                    size_t in_idx = ((size_t)b * (size_t)Ci + ci) * (size_t)(D * H * W) +
                                    (size_t)iz * (H * W) + 
                                    (size_t)iy * (size_t)W + ix;
                    size_t w_idx = ((size_t)co * (size_t)Ci + ci) * (size_t)(kd * kh * kw) +
                                   (size_t)kz * (kh * kw) + 
                                   (size_t)ky * (size_t)kw + kx;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }
    size_t out_idx = ((size_t)b * (size_t)Co + co) * (size_t)(Do * Ho * Wo) +
                     (size_t)oz * (Ho * Wo) + 
                     (size_t)oy * (size_t)Wo + ox;
    output[out_idx] = sum;
}

torch::Tensor conv3d_hip(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias
) {
    auto in_sizes = input.sizes();
    int64_t B = in_sizes[0];
    int64_t Ci = in_sizes[1];
    int64_t D = in_sizes[2];
    int64_t H = in_sizes[3];
    int64_t W = in_sizes[4];
    auto w_sizes = weight.sizes();
    int64_t Co = w_sizes[0];
    int64_t kd = w_sizes[2];
    int64_t kh = w_sizes[3];
    int64_t kw = w_sizes[4];
    int64_t Do = D - kd + 1;
    int64_t Ho = H - kh + 1;
    int64_t Wo = W - kw + 1;
    torch::Tensor out = torch::empty({B, Co, Do, Ho, Wo}, input.options());
    int64_t total = B * Co * Do * Ho * Wo;
    const int threads = 256;
    int64_t num_blocks_ = (total + threads - 1) / threads;
    unsigned int num_blocks = (unsigned int) num_blocks_;
    dim3 grid(num_blocks);
    dim3 blk(threads);
    hipLaunchKernelGGL(conv3d_kernel, grid, blk, 0, 0,
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        (int)B, (int)Ci, (int)D, (int)H, (int)W,
        (int)Co, (int)kd, (int)kh, (int)kw,
        (int)Do, (int)Ho, (int)Wo);
    hipDeviceSynchronize();
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        printf("HIP error after launch: %s\\n", hipGetErrorString(err));
    }
    return out;
}
"""

conv3d_module = load_inline(
    name="conv3d_custom",
    cpp_sources=cpp_source,
    functions=["conv3d_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        kd = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kd, kd, kd))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(self.bias)
        self.pool1 = nn.MaxPool3d(pool_kernel_size)
        self.pool2 = nn.MaxPool3d(pool_kernel_size)
        self.conv_hip = conv3d_module

    def forward(self, x):
        x = self.conv_hip.conv3d_hip(x, self.weight, self.bias)
        x = torch.softmax(x, dim=1)
        x = self.pool1(x)
        x = self.pool2(x)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
