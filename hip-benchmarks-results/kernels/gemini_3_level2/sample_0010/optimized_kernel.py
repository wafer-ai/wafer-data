
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>
#include <cmath>

__global__ void fused_pool_activation_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float scaling,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int pool_k, int pool_stride) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H_out * W_out;
    
    if (idx >= total_elements) return;
    
    // Decode index
    // Layout is N, C, H_out, W_out (contiguous)
    int w_out = idx % W_out;
    int tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp = tmp / H_out;
    int c = tmp % C;
    int n = tmp / C;
    
    int h_start = h_out * pool_stride;
    int w_start = w_out * pool_stride;
    
    const float* input_slice = input + (n * C + c) * (H_in * W_in);
    
    float max_val = -INFINITY;
    
    for (int i = 0; i < pool_k; ++i) {
        int r = h_start + i;
        for (int j = 0; j < pool_k; ++j) {
            int c_in = w_start + j;
            float val = input_slice[r * W_in + c_in];
            if (val > max_val) {
                max_val = val;
            }
        }
    }
    
    // Apply Tanh -> Scale -> Bias
    float res = tanhf(max_val);
    res = res * scaling;
    res = res + bias[c]; 
    
    output[idx] = res;
}

torch::Tensor fused_forward(torch::Tensor input, torch::Tensor bias, float scaling, 
                           int kernel_size, int stride) {
    int N = input.size(0);
    int C = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    
    int H_out = (H_in - kernel_size) / stride + 1;
    int W_out = (W_in - kernel_size) / stride + 1;
    
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::empty({N, C, H_out, W_out}, options);
    
    int total_elements = N * C * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_pool_activation_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling,
        N, C, H_in, W_in,
        H_out, W_out,
        kernel_size, stride
    );
    
    return output;
}
"""

fused_ops = load_inline(
    name="fused_ops_v3",
    cpp_sources=cpp_source,
    functions=["fused_forward"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_kernel_size 
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        if not x.is_contiguous():
            x = x.contiguous()
        return self.fused_ops.fused_forward(
            x, 
            self.bias.view(-1).contiguous(), 
            self.scaling_factor, 
            self.pool_kernel_size, 
            self.pool_stride
        )
