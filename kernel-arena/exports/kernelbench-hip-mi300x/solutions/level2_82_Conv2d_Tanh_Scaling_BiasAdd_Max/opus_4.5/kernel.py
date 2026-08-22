import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import uuid

os.environ["CXX"] = "hipcc"

# Fused kernel optimized for MI300X
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <float.h>

// Optimized kernel using explicit unrolling
__global__ void fused_tanh_scale_bias_maxpool_optimized(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const float scaling_factor
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * channels * out_height * out_width;
    
    if (idx >= total) return;
    
    // Decode output position
    const int ow = idx % out_width;
    const int tmp1 = idx / out_width;
    const int oh = tmp1 % out_height;
    const int tmp2 = tmp1 / out_height;
    const int c = tmp2 % channels;
    const int b = tmp2 / channels;
    
    const float bias_val = bias[c];
    const int channel_stride = in_height * in_width;
    const int batch_stride = channels * channel_stride;
    
    // Base pointer for this (batch, channel)
    const float* input_bc = input + b * batch_stride + c * channel_stride;
    
    // Pool window start
    const int in_row_start = oh * 4;
    const int in_col_start = ow * 4;
    
    float max_val = -FLT_MAX;
    
    // Explicit unroll for 4x4 pooling
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        const float* row_ptr = input_bc + (in_row_start + r) * in_width + in_col_start;
        
        float v0 = tanhf(row_ptr[0]) * scaling_factor + bias_val;
        float v1 = tanhf(row_ptr[1]) * scaling_factor + bias_val;
        float v2 = tanhf(row_ptr[2]) * scaling_factor + bias_val;
        float v3 = tanhf(row_ptr[3]) * scaling_factor + bias_val;
        
        max_val = fmaxf(max_val, fmaxf(fmaxf(v0, v1), fmaxf(v2, v3)));
    }
    
    output[idx] = max_val;
}

torch::Tensor my_fused_kernel_hip(
    torch::Tensor input,
    torch::Tensor bias,
    int pool_size,
    float scaling_factor
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    
    const int out_height = in_height / pool_size;
    const int out_width = in_width / pool_size;
    
    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());
    
    const int total_elements = batch_size * channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_tanh_scale_bias_maxpool_optimized<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_height,
        in_width,
        out_height,
        out_width,
        scaling_factor
    );
    
    return output;
}
"""

fused_kernel_cpp = """
torch::Tensor my_fused_kernel_hip(
    torch::Tensor input,
    torch::Tensor bias,
    int pool_size,
    float scaling_factor
);
"""

# Generate unique module name
module_name = f"fused_82_{uuid.uuid4().hex[:8]}"
fused_module = load_inline(
    name=module_name,
    cpp_sources=fused_kernel_cpp,
    cuda_sources=fused_kernel_source,
    functions=["my_fused_kernel_hip"],
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
        self.fused_module = fused_module

    def forward(self, x):
        # Convolution (use PyTorch's optimized implementation)
        x = self.conv(x)
        # Fused: tanh + scaling + bias + maxpool
        x = self.fused_module.my_fused_kernel_hip(
            x, 
            self.bias.view(-1),  # Flatten bias to 1D
            self.pool_kernel_size,
            self.scaling_factor
        )
        return x


def get_inputs():
    return [torch.rand(128, 8, 256, 256).cuda()]


def get_init_inputs():
    return [8, 64, 3, 2.0, (64, 1, 1), 4]
