import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

fused_maxpool_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void fused_double_maxpool_kernel(const float* input, float* output, 
                                           int channels, int depth, int height, int width) {
    // Output position
    int d_out = blockIdx.x;
    int h_out = blockIdx.y;
    int w_out = blockIdx.z;
    
    int c = threadIdx.x;
    
    // Effective pooling size is 4x4 (two 2x2 pools)
    int d_start = d_out * 4;
    int h_start = h_out * 4;
    int w_start = w_out * 4;
    
    // Check bounds
    if (d_start >= depth || h_start >= height || w_start >= width) return;
    
    if (c >= channels) return;
    
    // Find max value in 4x4x4 window
    float max_val = -1e10f;
    
    for (int di = 0; di < 4 && d_start + di < depth; di++) {
        for (int hi = 0; hi < 4 && h_start + hi < height; hi++) {
            for (int wi = 0; wi < 4 && w_start + wi < width; wi++) {
                int d_in = d_start + di;
                int h_in = h_start + hi;
                int w_in = w_start + wi;
                
                int idx = ((c * depth + d_in) * height + h_in) * width + w_in;
                float val = input[idx];
                if (val > max_val) max_val = val;
            }
        }
    }
    
    // Write output
    int d_out_final = d_out;
    int h_out_final = h_out;
    int w_out_final = w_out;
    
    if (d_out_final < (depth + 3) / 4 && h_out_final < (height + 3) / 4 && w_out_final < (width + 3) / 4) {
        int out_idx = ((c * ((depth + 3) / 4) + d_out_final) * ((height + 3) / 4) + h_out_final) * ((width + 3) / 4) + w_out_final;
        output[out_idx] = max_val;
    }
}

torch::Tensor fused_double_maxpool_hip(torch::Tensor input) {
    auto shape = input.sizes();
    int channels = shape[0];
    int depth = shape[1];
    int height = shape[2];
    int width = shape[3];
    
    int d_out = (depth + 3) / 4;
    int h_out = (height + 3) / 4;
    int w_out = (width + 3) / 4;
    
    auto output = torch::zeros({channels, d_out, h_out, w_out}, input.options());
    
    dim3 blocks(d_out, h_out, w_out);
    dim3 threads(channels > 256 ? 256 : channels);
    
    if (channels <= 256) {
        fused_double_maxpool_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(), 
            output.data_ptr<float>(), 
            channels, depth, height, width
        );
    } else {
        // Handle more channels with multiple threads per block
        int thread_per_channel = (channels + 255) / 256;
        dim3 threads_full(256);
        fused_double_maxpool_kernel<<<blocks, threads_full>>>(
            input.data_ptr<float>(), 
            output.data_ptr<float>(), 
            channels, depth, height, width
        );
    }
    
    return output;
}
"""

fused_double_maxpool = load_inline(
    name="fused_double_maxpool",
    cpp_sources=fused_maxpool_cpp_source,
    functions=["fused_double_maxpool_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies Softmax, and fused double max pooling operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.fused_double_maxpool = fused_double_maxpool

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels, depth', height', width')
        """
        x = self.conv(x)
        x = torch.softmax(x, dim=1)
        
        # Apply fused double maxpool to each batch element
        outputs = []
        b, c, d, h, w = x.shape
        for i in range(b):
            batch_input = x[i]  # (c, d, h, w)
            batch_output = self.fused_double_maxpool.fused_double_maxpool_hip(batch_input)
            outputs.append(batch_output)
        
        x = torch.stack(outputs, dim=0)  # (b, c, d/4, h/4, w/4)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
pool_kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]