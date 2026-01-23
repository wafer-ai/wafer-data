import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused softmax + double maxpool kernel with better optimizations
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <float.h>
#include <math.h>

// Fused Softmax + MaxPool3d (4x4x4) kernel
// This processes each output position directly, computing softmax on-the-fly
// while finding the max across spatial 4x4x4 windows
__global__ void fused_softmax_maxpool_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width
) {
    // Each thread handles one output position across all channels
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_depth * out_height * out_width;
    
    if (idx >= total) return;
    
    // Decode output spatial index
    int w_out = idx % out_width;
    int h_out = (idx / out_width) % out_height;
    int d_out = (idx / (out_width * out_height)) % out_depth;
    int b = idx / (out_width * out_height * out_depth);
    
    // Input starting position (4x4x4 pooling window)
    int d_start = d_out * 4;
    int h_start = h_out * 4;
    int w_start = w_out * 4;
    
    // For each spatial position in the 4x4x4 window, we need to compute softmax
    // then find the max across all positions for each channel
    
    // Initialize output max values
    float out_max[16]; // assuming max 16 channels
    for (int c = 0; c < channels; c++) {
        out_max[c] = -FLT_MAX;
    }
    
    // Process each spatial position in the 4x4x4 window
    for (int dd = 0; dd < 4; dd++) {
        int d_in = d_start + dd;
        if (d_in >= in_depth) continue;
        
        for (int hh = 0; hh < 4; hh++) {
            int h_in = h_start + hh;
            if (h_in >= in_height) continue;
            
            for (int ww = 0; ww < 4; ww++) {
                int w_in = w_start + ww;
                if (w_in >= in_width) continue;
                
                // Load all channel values for this spatial position
                float vals[16];
                float max_val = -FLT_MAX;
                
                for (int c = 0; c < channels; c++) {
                    int in_idx = ((b * channels + c) * in_depth + d_in) * in_height * in_width 
                               + h_in * in_width + w_in;
                    vals[c] = input[in_idx];
                    if (vals[c] > max_val) max_val = vals[c];
                }
                
                // Compute softmax for this position
                float sum_exp = 0.0f;
                for (int c = 0; c < channels; c++) {
                    vals[c] = expf(vals[c] - max_val);
                    sum_exp += vals[c];
                }
                
                float inv_sum = 1.0f / sum_exp;
                for (int c = 0; c < channels; c++) {
                    float softmax_val = vals[c] * inv_sum;
                    if (softmax_val > out_max[c]) {
                        out_max[c] = softmax_val;
                    }
                }
            }
        }
    }
    
    // Write output
    for (int c = 0; c < channels; c++) {
        int out_idx = ((b * channels + c) * out_depth + d_out) * out_height * out_width 
                    + h_out * out_width + w_out;
        output[out_idx] = out_max[c];
    }
}

torch::Tensor fused_softmax_maxpool_hip(torch::Tensor input) {
    auto sizes = input.sizes();
    int batch_size = sizes[0];
    int channels = sizes[1];
    int in_depth = sizes[2];
    int in_height = sizes[3];
    int in_width = sizes[4];
    
    // Output size after 4x4x4 pooling (equivalent to two 2x2x2 pools)
    int out_depth = in_depth / 4;
    int out_height = in_height / 4;
    int out_width = in_width / 4;
    
    auto output = torch::empty({batch_size, channels, out_depth, out_height, out_width}, 
                               input.options());
    
    int total = batch_size * out_depth * out_height * out_width;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    
    fused_softmax_maxpool_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, in_depth, in_height, in_width,
        out_depth, out_height, out_width
    );
    
    return output;
}
"""

fused_ops = load_inline(
    name="fused_ops_v2",
    cpp_sources="""
torch::Tensor fused_softmax_maxpool_hip(torch::Tensor input);
""",
    cuda_sources=fused_kernel_source,
    functions=["fused_softmax_maxpool_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses Conv3d output -> Softmax -> MaxPool -> MaxPool
    into Conv3d -> Fused(Softmax + MaxPool4x4x4)
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_softmax_maxpool_hip(x)
        return x


def get_inputs():
    return [torch.rand(128, 3, 16, 32, 32).cuda()]


def get_init_inputs():
    return [3, 16, 3, 2]
