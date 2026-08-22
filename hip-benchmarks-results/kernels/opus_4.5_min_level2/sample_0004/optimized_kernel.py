import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel: subtract1 -> tanh -> subtract2 -> avgpool
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

__global__ void fused_subtract_tanh_subtract_avgpool_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int pool_size,
    const float subtract1_value,
    const float subtract2_value
) {
    // Each thread computes one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    // Compute output coordinates
    int ow = idx % out_width;
    int oh = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int b = idx / (out_width * out_height * channels);
    
    // Compute input starting position for this pooling window
    int ih_start = oh * pool_size;
    int iw_start = ow * pool_size;
    
    // Compute average over pooling window with fused operations
    float sum = 0.0f;
    int count = 0;
    
    for (int ph = 0; ph < pool_size; ph++) {
        for (int pw = 0; pw < pool_size; pw++) {
            int ih = ih_start + ph;
            int iw = iw_start + pw;
            
            if (ih < in_height && iw < in_width) {
                int in_idx = b * channels * in_height * in_width + 
                             c * in_height * in_width + 
                             ih * in_width + iw;
                
                float val = input[in_idx];
                // Fused: subtract1 -> tanh -> subtract2
                val = val - subtract1_value;
                val = tanhf(val);
                val = val - subtract2_value;
                sum += val;
                count++;
            }
        }
    }
    
    output[idx] = sum / (float)count;
}

torch::Tensor fused_subtract_tanh_subtract_avgpool(
    torch::Tensor input,
    float subtract1_value,
    float subtract2_value,
    int pool_size
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    
    const int out_height = in_height / pool_size;
    const int out_width = in_width / pool_size;
    
    auto output = torch::empty({batch_size, channels, out_height, out_width}, 
                               input.options());
    
    const int total_elements = batch_size * channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_subtract_tanh_subtract_avgpool_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, in_height, in_width,
        out_height, out_width, pool_size,
        subtract1_value, subtract2_value
    );
    
    return output;
}
"""

fused_kernel_cpp = """
torch::Tensor fused_subtract_tanh_subtract_avgpool(
    torch::Tensor input,
    float subtract1_value,
    float subtract2_value,
    int pool_size
);
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_kernel_cpp,
    cuda_sources=fused_kernel_source,
    functions=["fused_subtract_tanh_subtract_avgpool"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses subtract, tanh, subtract, and avgpool operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool

    def forward(self, x):
        x = self.conv(x)
        x = fused_ops.fused_subtract_tanh_subtract_avgpool(
            x, self.subtract1_value, self.subtract2_value, self.kernel_size_pool
        )
        return x


def get_inputs():
    return [torch.rand(128, 64, 128, 128).cuda()]


def get_init_inputs():
    return [64, 128, 3, 0.5, 0.2, 2]
