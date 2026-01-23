import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

bn_scale_fused_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void bn_scale_fused_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ A,
    const float* __restrict__ B,
    int C, int H, int W) {
    
    int HW = H * W;
    int NCHW = C * HW;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = NCHW;
    
    if (idx < total) {
        int n = idx / NCHW;
        int c = (idx % NCHW) / HW;
        int hw = idx % HW;
        
        int linear_idx = n * NCHW + c * HW + hw;
        
        output[linear_idx] = input[linear_idx] * A[c] + B[c];
    }
}

torch::Tensor bn_scale_fused_hip(
    torch::Tensor input,
    torch::Tensor A,
    torch::Tensor B) {
    
    auto output = torch::zeros_like(input);
    
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    int total_elements = N * C * H * W;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    bn_scale_fused_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C, H, W);
    
    return output;
}
"""

bn_scale_fused = load_inline(
    name="bn_scale_fused",
    cpp_sources=bn_scale_fused_cpp_source,
    functions=["bn_scale_fused_hip"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor
        self.bn_scale_fused = bn_scale_fused
        
    def forward(self, x):
        x = self.conv(x)
        
        # Precompute BatchNorm + Scale parameters: y = x * A' + B'
        # where A' = gamma / sqrt(var + eps) * scaling_factor
        # and B' = (beta - gamma * mean / sqrt(var + eps)) * scaling_factor
        with torch.no_grad():
            inv_std = 1.0 / torch.sqrt(self.bn.running_var + self.bn.eps)
            A = self.bn.weight * inv_std * self.scaling_factor
            B = (self.bn.bias - self.bn.weight * self.bn.running_mean * inv_std) * self.scaling_factor
        
        x = self.bn_scale_fused.bn_scale_fused_hip(x, A, B)
        return x