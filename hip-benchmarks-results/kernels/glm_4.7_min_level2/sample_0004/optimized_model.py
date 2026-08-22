import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

fused_kernel_cpp_source_v6 = """
#include <hip/hip_runtime.h>
#include <math.h>

__global__ void fused_subtract_tanh_subtract_avgpool_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    float subtract1_value,
    float subtract2_value,
    int pool_size) {
    
    // Output dimensions after pooling
    int out_h = height / pool_size;
    int out_w = width / pool_size;
    
    // Total number of output pixels per batch
    int out_pixels_per_batch = channels * out_h * out_w;
    int total_out_pixels = batch_size * out_pixels_per_batch;
    
    // Process 4 output pixels per thread
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    for (int i = 0; i < 4 && idx < total_out_pixels; i++, idx++) {
        // Decompose linear index to batch, channel, out_h, out_w
        int batch = idx / out_pixels_per_batch;
        int remaining = idx % out_pixels_per_batch;
        int channel = remaining / (out_h * out_w);
        int out_row = (remaining % (out_h * out_w)) / out_w;
        int out_col = remaining % out_w;
        
        // Calculate base input row and column
        int base_row = out_row * pool_size;
        int base_col = out_col * pool_size;
        
        // Pointer to the start of this batch-channel's input data
        const float* batch_channel_data = input + (batch * channels + channel) * height * width;
        
        // For average pooling, compute average over pool_size x pool_size window
        float sum = 0.0f;
        float pool_area = (float)(pool_size * pool_size);
        
        for (int ph = 0; ph < pool_size; ph++) {
            int h = base_row + ph;
            const float* row_data = batch_channel_data + h * width;
            
            for (int pw = 0; pw < pool_size; pw++) {
                int w = base_col + pw;
                
                // Load value
                float val = row_data[w];
                
                // Apply fused operations: subtract -> tanh -> subtract
                val = tanhf(val - subtract1_value);
                sum += val - subtract2_value;
            }
        }
        
        // Compute output index and store result
        int out_idx = ((batch * channels + channel) * out_h + out_row) * out_w + out_col;
        output[out_idx] = sum / pool_area;
    }
}

torch::Tensor fused_subtract_tanh_subtract_avgpool(
    torch::Tensor input,
    float subtract1_value,
    float subtract2_value,
    int pool_size) {
    
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    int out_h = height / pool_size;
    int out_w = width / pool_size;
    int total_out_pixels = batch_size * channels * out_h * out_w;
    
    auto output = torch::zeros({batch_size, channels, out_h, out_w}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (total_out_pixels + 3) / 4 / block_size + 1;
    
    fused_subtract_tanh_subtract_avgpool_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width,
        subtract1_value,
        subtract2_value,
        pool_size);
    
    return output;
}
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_kernel_cpp_source_v6,
    functions=["fused_subtract_tanh_subtract_avgpool"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Model that performs a convolution, subtraction, tanh activation, subtraction and average pooling.
    Optimized version with fused elementwise and pooling operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        # Fused kernel: subtract, tanh, subtract, avgpool
        x = self.fused_ops.fused_subtract_tanh_subtract_avgpool(
            x, self.subtract1_value, self.subtract2_value, self.kernel_size_pool
        )
        return x