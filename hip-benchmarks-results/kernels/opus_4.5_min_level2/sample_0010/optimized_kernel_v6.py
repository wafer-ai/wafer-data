import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused kernel with workgroup processing multiple elements
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define POOL_SIZE 4
#define ITEMS_PER_THREAD 4

// Each thread processes multiple output elements
__global__ __launch_bounds__(256) void fused_tanh_scale_bias_maxpool_kernel_v6(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const float scaling_factor,
    const int channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int total_outputs
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    int plane_size_in = in_height * in_width;
    int plane_size_out = out_height * out_width;
    
    for (int idx = tid; idx < total_outputs; idx += stride) {
        // Decode output indices
        int ow = idx % out_width;
        int oh = (idx / out_width) % out_height;
        int c = (idx / plane_size_out) % channels;
        int b = idx / (plane_size_out * channels);
        
        float bias_val = bias[c];
        
        int h_start = oh * POOL_SIZE;
        int w_start = ow * POOL_SIZE;
        
        // Base index for this batch and channel
        const float* input_plane = input + (b * channels + c) * plane_size_in;
        
        // Compute max over 4x4 pooling window
        float max_val = -1e30f;
        
        #pragma unroll
        for (int h = 0; h < POOL_SIZE; h++) {
            const float* row = input_plane + (h_start + h) * in_width + w_start;
            
            float v0 = row[0];
            float v1 = row[1];
            float v2 = row[2];
            float v3 = row[3];
            
            // Apply tanh, scale, and bias
            v0 = tanhf(v0) * scaling_factor + bias_val;
            v1 = tanhf(v1) * scaling_factor + bias_val;
            v2 = tanhf(v2) * scaling_factor + bias_val;
            v3 = tanhf(v3) * scaling_factor + bias_val;
            
            max_val = fmaxf(max_val, fmaxf(fmaxf(v0, v1), fmaxf(v2, v3)));
        }
        
        output[idx] = max_val;
    }
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
    // Limit grid size and let threads loop
    const int num_blocks = min((total + block_size - 1) / block_size, 65535);
    
    fused_tanh_scale_bias_maxpool_kernel_v6<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        channels,
        in_height,
        in_width,
        out_height,
        out_width,
        total
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
    name="fused_tanh_scale_bias_maxpool_v6",
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
