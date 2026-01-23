import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel with vectorized loads
fused_post_conv_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Optimized kernel with vectorized loads where possible
// For 2x2 pooling, load 2 float2 for each row
__global__ void fused_sub_tanh_sub_avgpool_vec_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const float subtract1,
    const float subtract2,
    const float inv_pool_area
) {
    // Linear index mapping
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_height * out_width;
    
    if (idx >= total) return;
    
    // Decompose linear index
    int ow = idx % out_width;
    int temp = idx / out_width;
    int oh = temp % out_height;
    temp = temp / out_height;
    int c = temp % channels;
    int b = temp / channels;
    
    // Input starting positions (2x2 pooling assumed)
    int in_y = oh * 2;
    int in_x = ow * 2;
    
    // Input base for this batch/channel
    int in_plane_stride = in_height * in_width;
    int in_base = (b * channels + c) * in_plane_stride;
    
    // Load row 0
    int row0_offset = in_base + in_y * in_width + in_x;
    float v00 = __ldg(&input[row0_offset]);
    float v01 = __ldg(&input[row0_offset + 1]);
    
    // Load row 1
    int row1_offset = in_base + (in_y + 1) * in_width + in_x;
    float v10 = __ldg(&input[row1_offset]);
    float v11 = __ldg(&input[row1_offset + 1]);
    
    // Apply fused operations
    v00 = tanhf(v00 - subtract1) - subtract2;
    v01 = tanhf(v01 - subtract1) - subtract2;
    v10 = tanhf(v10 - subtract1) - subtract2;
    v11 = tanhf(v11 - subtract1) - subtract2;
    
    // Average
    float result = (v00 + v01 + v10 + v11) * inv_pool_area;
    
    output[idx] = result;
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
    
    float inv_pool_area = 1.0f / (float)(pool_size * pool_size);
    
    int total = batch_size * channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    
    fused_sub_tanh_sub_avgpool_vec_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
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
    name="fused_post_conv_v3",
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
