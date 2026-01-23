import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

fused_softmax_maxpool_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void fused_softmax_maxpool_kernel(const float* input, float* output, int out_channels, int depth, int height, int width) {
    int d_out = blockIdx.x;
    int h_out = blockIdx.y;
    int w_out = blockIdx.z;
    
    int d_start = d_out * 2;
    int h_start = h_out * 2;
    int w_start = w_out * 2;
    
    // Check bounds
    if (d_start >= depth || h_start >= height || w_start >= width) return;
    
    // Find which pixel in the 2x2 window has the maximum softmax value
    float max_softmax_val = -1.0f;
    int best_c = 0;
    int best_di = 0;
    int best_hi = 0;
    int best_wi = 0;
    
    for (int di = 0; di < 2 && d_start + di < depth; di++) {
        for (int hi = 0; hi < 2 && h_start + hi < height; hi++) {
            for (int wi = 0; wi < 2 && w_start + wi < width; wi++) {
                int d_in = d_start + di;
                int h_in = h_start + hi;
                int w_in = w_start + wi;
                
                // Find max value for softmax normalization
                float max_val = -1e10f;
                for (int c = 0; c < out_channels; c++) {
                    int idx = ((c * depth + d_in) * height + h_in) * width + w_in;
                    float val = input[idx];
                    if (val > max_val) max_val = val;
                }
                
                // Compute sum of exponentials
                float exp_sum = 0.0f;
                for (int c = 0; c < out_channels; c++) {
                    int idx = ((c * depth + d_in) * height + h_in) * width + w_in;
                    exp_sum += expf(input[idx] - max_val);
                }
                
                if (exp_sum <= 0.0f) exp_sum = 1.0f;
                
                // Find channel with maximum softmax value
                float max_exp_val = -1.0f;
                int local_best_c = 0;
                
                for (int c = 0; c < out_channels; c++) {
                    int idx = ((c * depth + d_in) * height + h_in) * width + w_in;
                    float exp_val = expf(input[idx] - max_val);
                    if (exp_val > max_exp_val) {
                        max_exp_val = exp_val;
                        local_best_c = c;
                    }
                }
                
                float softmax_val = max_exp_val / exp_sum;
                
                if (softmax_val > max_softmax_val) {
                    max_softmax_val = softmax_val;
                    best_c = local_best_c;
                    best_di = di;
                    best_hi = hi;
                    best_wi = wi;
                }
            }
        }
    }
    
    if (max_softmax_val >= 0.0f) {
        int d_in = d_start + best_di;
        int h_in = h_start + best_hi;
        int w_in = w_start + best_wi;
        
        int out_idx = ((best_c * (depth/2) + d_out) * (height/2) + h_out) * (width/2) + w_out;
        int in_idx = ((best_c * depth + d_in) * height + h_in) * width + w_in;
        output[out_idx] = input[in_idx];
    }
}

torch::Tensor fused_softmax_maxpool_hip(torch::Tensor input) {
    auto shape = input.sizes();
    int out_channels = shape[0];
    int depth = shape[1];
    int height = shape[2];
    int width = shape[3];
    
    int d_out = (depth + 1) / 2;
    int h_out = (height + 1) / 2;
    int w_out = (width + 1) / 2;
    
    auto output = torch::zeros({out_channels, d_out, h_out, w_out}, input.options());
    
    dim3 blocks(d_out, h_out, w_out);
    
    fused_softmax_maxpool_kernel<<<blocks, 1>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        out_channels, depth, height, width
    );
    
    return output;
}
"""

fused_softmax_maxpool = load_inline(
    name="fused_softmax_maxpool",
    cpp_sources=fused_softmax_maxpool_cpp_source,
    functions=["fused_softmax_maxpool_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies fused Softmax+MaxPool, and a second max pooling operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.fused_softmax_maxpool = fused_softmax_maxpool
        self.pool2 = nn.MaxPool3d(pool_kernel_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels, depth', height', width')
        """
        x = self.conv(x)
        
        # Apply fused softmax+maxpool to each batch element
        outputs = []
        b, c, d, h, w = x.shape
        for i in range(b):
            batch_input = x[i]  # (c, d, h, w)
            batch_output = self.fused_softmax_maxpool.fused_softmax_maxpool_hip(batch_input)
            outputs.append(batch_output)
        
        x = torch.stack(outputs, dim=0)  # (b, c, d/2, h/2, w/2)
        
        # Second pool
        x = self.pool2(x)
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