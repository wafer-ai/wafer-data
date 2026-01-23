import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel for: subtract1 -> tanh -> subtract2 -> avgpool
fused_post_conv_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__global__ void fused_sub_tanh_sub_avgpool_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int pool_size,
    const float subtract1,
    const float subtract2
) {
    // Each thread computes one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_height * out_width;
    
    if (idx >= total) return;
    
    // Decompose idx to batch, channel, output y, output x
    int ow = idx % out_width;
    int oh = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int b = idx / (out_width * out_height * channels);
    
    // Starting position in input
    int in_y_start = oh * pool_size;
    int in_x_start = ow * pool_size;
    
    float sum = 0.0f;
    float pool_area = (float)(pool_size * pool_size);
    
    // Apply operations and pool
    for (int py = 0; py < pool_size; py++) {
        for (int px = 0; px < pool_size; px++) {
            int in_y = in_y_start + py;
            int in_x = in_x_start + px;
            
            int in_idx = b * (channels * in_height * in_width) + 
                         c * (in_height * in_width) + 
                         in_y * in_width + in_x;
            
            float val = input[in_idx];
            val = val - subtract1;       // subtract1
            val = tanhf(val);            // tanh
            val = val - subtract2;       // subtract2
            sum += val;
        }
    }
    
    output[idx] = sum / pool_area;
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
    
    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());
    
    int total = batch_size * channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    
    fused_sub_tanh_sub_avgpool_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_height,
        in_width,
        out_height,
        out_width,
        pool_size,
        subtract1,
        subtract2
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
    name="fused_post_conv",
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
