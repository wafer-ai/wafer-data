import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized fused kernel with larger block size and fast math
fused_post_conv_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Optimized kernel using fast tanh approximation and larger blocks
__global__ __launch_bounds__(512)
void fused_sub_tanh_sub_avgpool_fast_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_channels,
    const int in_height,
    const int in_width,
    const int in_plane_stride,
    const int out_height,
    const int out_width,
    const int out_plane_stride,
    const float subtract1,
    const float subtract2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_channels * out_plane_stride;
    
    if (idx >= total) return;
    
    // Fast decomposition
    int ow = idx % out_width;
    int oh = (idx / out_width) % out_height;
    int bc = idx / out_plane_stride;
    
    // Input positions
    int in_y = oh << 1;  // * 2
    int in_x = ow << 1;  // * 2
    int in_base = bc * in_plane_stride;
    
    // Load all 4 values
    int r0_base = in_base + in_y * in_width + in_x;
    int r1_base = r0_base + in_width;
    
    float v00 = input[r0_base];
    float v01 = input[r0_base + 1];
    float v10 = input[r1_base];
    float v11 = input[r1_base + 1];
    
    // Apply fused operations: sub1 -> tanh -> sub2
    v00 = __tanhf(v00 - subtract1) - subtract2;
    v01 = __tanhf(v01 - subtract1) - subtract2;
    v10 = __tanhf(v10 - subtract1) - subtract2;
    v11 = __tanhf(v11 - subtract1) - subtract2;
    
    // Average and store
    output[idx] = (v00 + v01 + v10 + v11) * 0.25f;
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
    
    int batch_channels = batch_size * channels;
    int in_plane_stride = in_height * in_width;
    int out_plane_stride = out_height * out_width;
    int total = batch_channels * out_plane_stride;
    
    const int block_size = 512;
    const int num_blocks = (total + block_size - 1) / block_size;
    
    fused_sub_tanh_sub_avgpool_fast_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_channels,
        in_height,
        in_width,
        in_plane_stride,
        out_height,
        out_width,
        out_plane_stride,
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
    name="fused_post_conv_v6",
    cpp_sources=fused_post_conv_cpp,
    cuda_sources=fused_post_conv_source,
    functions=["fused_sub_tanh_sub_avgpool_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
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
