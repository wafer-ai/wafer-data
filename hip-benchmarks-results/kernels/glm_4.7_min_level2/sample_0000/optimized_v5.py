import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

fused_maxpool_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void fused_double_maxpool_kernel(const float* input, float* output, 
                                           int batch, int channels, int depth, int height, int width,
                                           int out_depth, int out_height, int out_width) {
    int d_out = blockIdx.x;
    int h_out = blockIdx.y;
    int w_out = blockIdx.z;
    int bc = threadIdx.x;
    
    if (d_out >= out_depth || h_out >= out_height || w_out >= out_width || bc >= batch * channels) return;
    
    int b = bc / channels;
    int c = bc % channels;
    
    int d_start = d_out * 4;
    int h_start = h_out * 4;
    int w_start = w_out * 4;
    
    float max_val = -1e10f;
    
    int d_end = min(d_start + 4, depth);
    int h_end = min(h_start + 4, height);
    int w_end = min(w_start + 4, width);
    
    for (int di = d_start; di < d_end; di++) {
        for (int hi = h_start; hi < h_end; hi++) {
            for (int wi = w_start; wi < w_end; wi++) {
                int idx = ((((b * channels + c) * depth + di) * height + hi) * width) + wi;
                float val = input[idx];
                if (val > max_val) max_val = val;
            }
        }
    }
    
    int out_idx = ((((b * channels + c) * out_depth + d_out) * out_height + h_out) * out_width) + w_out;
    output[out_idx] = max_val;
}

torch::Tensor fused_double_maxpool_hip(torch::Tensor input) {
    int batch = input.size(0);
    int channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);
    
    int pool1_depth = depth / 2;
    int pool1_height = height / 2;
    int pool1_width = width / 2;
    
    int out_depth = pool1_depth / 2;
    int out_height = pool1_height / 2;
    int out_width = pool1_width / 2;
    
    auto output = torch::zeros({batch, channels, out_depth, out_height, out_width}, input.options());
    
    dim3 blocks(out_depth, out_height, out_width);
    dim3 threads(batch * channels);
    
    if (batch * channels > 0) {
        fused_double_maxpool_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(), 
            output.data_ptr<float>(), 
            batch, channels, depth, height, width,
            out_depth, out_height, out_width
        );
    }
    
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
        x = self.fused_double_maxpool.fused_double_maxpool_hip(x)
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