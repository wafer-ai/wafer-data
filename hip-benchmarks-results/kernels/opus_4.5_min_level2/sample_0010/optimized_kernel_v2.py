import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused kernel for tanh + scaling + bias + maxpool with better memory access
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Optimized kernel using shared memory for bias and better thread organization
__global__ void fused_tanh_scale_bias_maxpool_kernel_v2(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const float scaling_factor,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int pool_size
) {
    // Shared memory for bias values
    __shared__ float s_bias[64];  // Assuming max 64 channels, adjust if needed
    
    // Load bias into shared memory
    if (threadIdx.x < channels) {
        s_bias[threadIdx.x] = bias[threadIdx.x];
    }
    __syncthreads();
    
    // Each thread computes one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_height * out_width;
    
    if (idx >= total) return;
    
    // Decode output indices
    int ow = idx % out_width;
    int oh = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int b = idx / (out_width * out_height * channels);
    
    // Compute max over pooling window
    float max_val = -INFINITY;
    
    int h_start = oh * pool_size;
    int w_start = ow * pool_size;
    
    float bias_val = s_bias[c];
    
    // Base index for this batch and channel
    int base_idx = (b * channels + c) * in_height * in_width;
    
    // Unroll for pool_size = 4
    #pragma unroll
    for (int h = 0; h < pool_size; h++) {
        int h_idx = h_start + h;
        if (h_idx >= in_height) continue;
        
        int row_base = base_idx + h_idx * in_width + w_start;
        
        #pragma unroll
        for (int w = 0; w < pool_size; w++) {
            int w_idx = w_start + w;
            if (w_idx >= in_width) continue;
            
            float val = input[row_base + w];
            // Fused: tanh, scaling, bias
            val = tanhf(val) * scaling_factor + bias_val;
            max_val = fmaxf(max_val, val);
        }
    }
    
    output[idx] = max_val;
}

torch::Tensor fused_tanh_scale_bias_maxpool_hip(
    torch::Tensor input,
    torch::Tensor bias,
    float scaling_factor,
    int pool_size
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    
    const int out_height = in_height / pool_size;
    const int out_width = in_width / pool_size;
    
    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());
    
    const int total = batch_size * channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    
    fused_tanh_scale_bias_maxpool_kernel_v2<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        batch_size,
        channels,
        in_height,
        in_width,
        out_height,
        out_width,
        pool_size
    );
    
    return output;
}
"""

fused_cpp_source = """
torch::Tensor fused_tanh_scale_bias_maxpool_hip(
    torch::Tensor input,
    torch::Tensor bias,
    float scaling_factor,
    int pool_size
);
"""

fused_module = load_inline(
    name="fused_tanh_scale_bias_maxpool_v2",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_tanh_scale_bias_maxpool_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
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
        self.fused_op = fused_module

    def forward(self, x):
        # Convolution (use PyTorch's optimized implementation)
        x = self.conv(x)
        # Fused: tanh + scaling + bias + maxpool
        x = self.fused_op.fused_tanh_scale_bias_maxpool_hip(
            x, 
            self.bias.view(-1),  # Flatten bias to 1D
            self.scaling_factor,
            self.pool_kernel_size
        )
        return x


def get_inputs():
    return [torch.rand(128, 8, 256, 256).cuda()]


def get_init_inputs():
    return [8, 64, 3, 2.0, (64, 1, 1), 4]
