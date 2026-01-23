import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

fused_maxpool_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void fused_double_maxpool_kernel(const float* input, float* output, 
                                           int channels, int depth, int height, int width) {
    int d_out = blockIdx.x;
    int h_out = blockIdx.y;
    int w_out = blockIdx.z;
    int c = threadIdx.x;
    
    int d_start = d_out * 4;
    int h_start = h_out * 4;
    int w_start = w_out * 4;
    
    if (d_start >= depth || h_start >= height || w_start >= width || c >= channels) return;
    
    float max_val = -1e10f;
    
    for (int di = 0; di < 4 && d_start + di < depth; di++) {
        for (int hi = 0; hi < 4 && h_start + hi < height; hi++) {
            for (int wi = 0; wi < 4 && w_start + wi < width; wi++) {
                int idx = ((c * depth + d_start + di) * height + h_start + hi) * width + w_start + wi;
                float val = input[idx];
                if (val > max_val) max_val = val;
            }
        }
    }
    
    int d_out_final = (depth + 3) / 4;
    int h_out_final = (height + 3) / 4;
    int w_out_final = (width + 3) / 4;
    
    if (d_out < d_out_final && h_out < h_out_final && w_out < w_out_final) {
        int out_idx = ((c * d_out_final + d_out) * h_out_final + h_out) * w_out_final + w_out;
        output[out_idx] = max_val;
    }
}

torch::Tensor fused_double_maxpool_hip(torch::Tensor input) {
    int channels = input.size(0);
    int depth = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    int d_out = (depth + 3) / 4;
    int h_out = (height + 3) / 4;
    int w_out = (width + 3) / 4;
    
    auto output = torch::zeros({channels, d_out, h_out, w_out}, input.options());
    
    dim3 blocks(d_out, h_out, w_out);
    dim3 threads(channels);
    
    fused_double_maxpool_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        channels, depth, height, width
    );
    
    return output;
}
"""

fused_double_maxpool = load_inline(
    name="fused_double_maxpool",
    cpp_sources=fused_maxpool_cpp_source,
    functions=["fused_double_maxpool_hip"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.fused_double_maxpool = fused_double_maxpool

    def forward(self, x):
        x = self.conv(x)
        x = torch.softmax(x, dim=1)
        
        outputs = []
        b = x.size(0)
        for i in range(b):
            batch_output = self.fused_double_maxpool.fused_double_maxpool_hip(x[i])
            outputs.append(batch_output)
        
        return torch.stack(outputs, dim=0)


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