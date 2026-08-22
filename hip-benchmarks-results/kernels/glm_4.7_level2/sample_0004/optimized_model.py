import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel: subtract -> tanh -> subtract -> avgpool
# Optimized with precomputed float division
fused_activ_pool_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void fused_activ_pool_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    int batch_size, int channels, int in_height, int in_width,
    int pool_kernel_size, float subtract1, float subtract2, float inv_pool_size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * (in_height / pool_kernel_size) * (in_width / pool_kernel_size);
    
    if (idx >= total_elements) return;
    
    int pool_height = in_height / pool_kernel_size;
    int pool_width = in_width / pool_kernel_size;
    int pool_size = pool_height * pool_width;
    
    int batch_channel = idx / pool_size;
    int pool_idx = idx % pool_size;
    
    int b = batch_channel / channels;
    int c = batch_channel % channels;
    int pool_row = pool_idx / pool_width;
    int pool_col = pool_idx % pool_width;
    
    // Compute average over the pool region
    int start_row = pool_row * pool_kernel_size;
    int start_col = pool_col * pool_kernel_size;
    
    float pool_sum = 0.0f;
    
    // Fixed count for pool_kernel_size=2 -> count is always 4
    for (int ph = 0; ph < pool_kernel_size; ph++) {
        int h = start_row + ph;
        if (h < in_height) {
            for (int pw = 0; pw < pool_kernel_size; pw++) {
                int w = start_col + pw;
                if (w < in_width) {
                    int input_idx = b * channels * in_height * in_width +
                                   c * in_height * in_width +
                                   h * in_width + w;
                    
                    float val = input[input_idx];
                    val = val - subtract1;
                    val = tanhf(val);
                    val = val - subtract2;
                    
                    pool_sum += val;
                }
            }
        }
    }
    
    output[idx] = pool_sum * inv_pool_size;
}

torch::Tensor fused_activ_pool_hip(
    torch::Tensor input,
    float subtract1, float subtract2, int pool_kernel_size) {
    
    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_height = in_height / pool_kernel_size;
    int out_width = in_width / pool_kernel_size;
    
    auto output = torch::zeros({batch_size, channels, out_height, out_width},
                               input.options());
    
    int total_elements = batch_size * channels * out_height * out_width;
    const int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    // Precompute 1.0f / (pool_kernel_size * pool_kernel_size) for division optimization
    float inv_pool_size = 1.0f / (float)(pool_kernel_size * pool_kernel_size);
    
    fused_activ_pool_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, in_height, in_width,
        pool_kernel_size, subtract1, subtract2, inv_pool_size);
    
    return output;
}
"""

fused_activ_pool = load_inline(
    name="fused_activ_pool",
    cpp_sources=fused_activ_pool_cpp_source,
    functions=["fused_activ_pool_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool
        self.fused_op = fused_activ_pool
        
    def forward(self, x):
        # Apply conv2d (keep PyTorch's optimized implementation)
        x = self.conv(x)
        # Apply fused subtract->tanh->subtract->avgpool
        x = self.fused_op.fused_activ_pool_hip(
            x, self.subtract1_value, self.subtract2_value, self.kernel_size_pool
        )
        return x