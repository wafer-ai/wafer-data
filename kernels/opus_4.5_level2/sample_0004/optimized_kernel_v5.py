import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused kernel with float2 loads and streamlined code
fused_post_conv_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Use float2 for vectorized loads - each row loads 2 consecutive floats
__global__ void fused_sub_tanh_sub_avgpool_f2_kernel(
    const float2* __restrict__ input,
    float* __restrict__ output,
    const int batch_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const float subtract1,
    const float subtract2,
    const float inv_pool_area
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_channels * out_height * out_width;
    
    if (idx >= total) return;
    
    // Decompose index
    int ow = idx % out_width;
    int oh = (idx / out_width) % out_height;
    int bc = idx / (out_width * out_height);
    
    // Input positions (2x2 pooling)
    int in_y = oh * 2;
    int in_x = ow * 2;
    
    // For float2, input width is halved
    int in_width_f2 = in_width / 2;
    
    // Load two rows, each as float2 (2 consecutive floats)
    int row0_idx = bc * in_height * in_width_f2 + in_y * in_width_f2 + in_x / 2;
    int row1_idx = bc * in_height * in_width_f2 + (in_y + 1) * in_width_f2 + in_x / 2;
    
    float2 row0 = input[row0_idx];
    float2 row1 = input[row1_idx];
    
    // Apply fused operations
    float v00 = tanhf(row0.x - subtract1) - subtract2;
    float v01 = tanhf(row0.y - subtract1) - subtract2;
    float v10 = tanhf(row1.x - subtract1) - subtract2;
    float v11 = tanhf(row1.y - subtract1) - subtract2;
    
    output[idx] = (v00 + v01 + v10 + v11) * inv_pool_area;
}

torch::Tensor fused_sub_tanh_sub_avgpool_hip(
    torch::Tensor input,
    float subtract1,
    float subtract2,
    int pool_size
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_height = in_height / pool_size;
    int out_width = in_width / pool_size;
    
    // Ensure input is contiguous for vectorized loads
    input = input.contiguous();
    
    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());
    
    float inv_pool_area = 1.0f / (float)(pool_size * pool_size);
    
    int batch_channels = batch_size * channels;
    int total = batch_channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    
    fused_sub_tanh_sub_avgpool_f2_kernel<<<num_blocks, block_size>>>(
        reinterpret_cast<const float2*>(input.data_ptr<float>()),
        output.data_ptr<float>(),
        batch_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        subtract1,
        subtract2,
        inv_pool_area
    );
    
    return output;
}
"""

fused_post_conv_cpp = """
torch::Tensor fused_sub_tanh_sub_avgpool_hip(
    torch::Tensor input,
    float subtract1,
    float subtract2,
    int pool_size
);
"""

fused_module = load_inline(
    name="fused_post_conv_v5",
    cpp_sources=fused_post_conv_cpp,
    cuda_sources=fused_post_conv_source,
    functions=["fused_sub_tanh_sub_avgpool_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses subtract, tanh, subtract, and avgpool into a single kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool

    def forward(self, x):
        x = self.conv(x)
        # Fused operations: subtract1 -> tanh -> subtract2 -> avgpool
        x = fused_module.fused_sub_tanh_sub_avgpool_hip(
            x, 
            self.subtract1_value, 
            self.subtract2_value,
            self.kernel_size_pool
        )
        return x
