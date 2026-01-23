
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

fused_ops_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__global__ void fused_ops_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int pool_size,
    float s1,
    float s2) {
    
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z;

    if (ow < out_w && oh < out_h) {
        int in_base = nc * in_h * in_w;
        float sum = 0.0f;
        for (int i = 0; i < pool_size; ++i) {
            int ih = oh * pool_size + i;
            int row_base = in_base + ih * in_w;
            for (int j = 0; j < pool_size; ++j) {
                int iw = ow * pool_size + j;
                sum += tanhf(input[row_base + iw] - s1);
            }
        }
        output[(nc * out_h + oh) * out_w + ow] = sum / (float)(pool_size * pool_size) - s2;
    }
}

torch::Tensor fused_ops_hip(
    torch::Tensor input,
    int pool_size,
    float s1,
    float s2) {
    
    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    
    int out_h = in_h / pool_size;
    int out_w = in_w / pool_size;
    
    auto output = torch::empty({batch_size, channels, out_h, out_w}, input.options());
    
    dim3 block_dim(32, 8);
    dim3 grid_dim((out_w + block_dim.x - 1) / block_dim.x, 
                   (out_h + block_dim.y - 1) / block_dim.y, 
                   batch_size * channels);
    
    fused_ops_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        in_h,
        in_w,
        out_h,
        out_w,
        pool_size,
        s1,
        s2
    );
    
    return output;
}
"""

fused_ops_module = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_kernel_source,
    functions=["fused_ops_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size).cuda()
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool
        self.fused_ops = fused_ops_module

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_ops_hip(x, self.kernel_size_pool, self.subtract1_value, self.subtract2_value)
        return x

def get_inputs():
    batch_size = 128
    in_channels = 64
    height, width = 128, 128
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    in_channels = 64
    out_channels = 128
    kernel_size = 3
    subtract1_value = 0.5
    subtract2_value = 0.2
    kernel_size_pool = 2
    return [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool]
