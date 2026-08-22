import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Custom HIP kernel source code
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <math.h>

#define INFINITY (__int_as_float(0x7f800000))

__global__ void fused_tanh_scale_bias_maxpool_kernel(
    const float* input,
    const float* bias,
    float* output,
    int N,
    int C,
    int H,
    int W,
    float scaling_factor,
    int pool_kernel_size,
    int H_out,
    int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_pixels = N * C * H_out * W_out;
    
    if (idx >= total_output_pixels) return;
    
    // Compute n, c, h_out, w_out from linear index
    int w_out = idx % W_out;
    int temp = idx / W_out;
    int h_out = temp % H_out;
    temp = temp / H_out;
    int c = temp % C;
    int n = temp / C;
    
    // Compute start position in input
    int h_start = h_out * pool_kernel_size;
    int w_start = w_out * pool_kernel_size;
    
    float max_val = -INFINITY;
    float bias_val = bias[c];
    
    // Iterate over the pooling window
    for (int i = 0; i < pool_kernel_size; i++) {
        int h_in = h_start + i;
        if (h_in >= H) continue;  // Bounds check
        
        for (int j = 0; j < pool_kernel_size; j++) {
            int w_in = w_start + j;
            if (w_in >= W) continue;  // Bounds check
            
            // Calculate input index for NCHW format
            int in_idx = ((n * C + c) * H + h_in) * W + w_in;
            float val = input[in_idx];
            
            // Apply tanh activation
            val = tanhf(val);
            // Apply scaling
            val = val * scaling_factor;
            // Apply bias (per-channel)
            val = val + bias_val;
            
            // Max pooling
            if (val > max_val) {
                max_val = val;
            }
        }
    }
    
    // Calculate output index for NCHW format
    int out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
    output[out_idx] = max_val;
}

torch::Tensor fused_tanh_scale_bias_maxpool(
    torch::Tensor input,
    torch::Tensor bias,
    float scaling_factor,
    int pool_kernel_size
) {
    // Input shape: (N, C, H, W)
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    
    // Output spatial dimensions
    int H_out = H / pool_kernel_size;
    int W_out = W / pool_kernel_size;
    
    // Create output tensor
    auto output = torch::empty({N, C, H_out, W_out}, input.options());
    
    // Launch kernel
    const int threads_per_block = 256;
    const int total_output_pixels = N * C * H_out * W_out;
    const int num_blocks = (total_output_pixels + threads_per_block - 1) / threads_per_block;
    
    fused_tanh_scale_bias_maxpool_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        scaling_factor,
        pool_kernel_size,
        H_out, W_out
    );
    
    return output;
}
"""

# Compile the custom kernel
fused_kernel = load_inline(
    name="fused_kernel",
    cpp_sources=fused_kernel_source,
    functions=["fused_tanh_scale_bias_maxpool"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses tanh, scaling, bias addition, and max-pooling into a single kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.pool_kernel_size = pool_kernel_size
        self.fused_kernel = fused_kernel

    def forward(self, x):
        # Convolution (using optimized PyTorch implementation)
        x = self.conv(x)
        
        # Fused tanh + scaling + bias + maxpool
        x = self.fused_kernel.fused_tanh_scale_bias_maxpool(
            x, self.bias, self.scaling_factor, self.pool_kernel_size
        )
        
        return x


def get_inputs():
    batch_size = 128
    in_channels = 8
    height, width = 256, 256
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    in_channels = 8
    out_channels = 64
    kernel_size = 3
    scaling_factor = 2.0
    bias_shape = (out_channels, 1, 1)
    pool_kernel_size = 4
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size]
