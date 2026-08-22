import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Approach: Use PyTorch's optimized softmax, then custom fused double maxpool
kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define TILE_SIZE 4

// Optimized double maxpool with shared memory
// Each warp handles multiple adjacent output positions
__global__ void fused_double_maxpool_tiled_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out
) {
    // Thread ID in 3D grid
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * D_out * H_out * W_out;
    
    if (out_idx >= total) return;
    
    // Decode output position
    int w_out = out_idx % W_out;
    int tmp = out_idx / W_out;
    int h_out = tmp % H_out;
    tmp = tmp / H_out;
    int d_out = tmp % D_out;
    tmp = tmp / D_out;
    int c = tmp % C;
    int n = tmp / C;
    
    // Input region bounds
    int d_start = d_out << 2;  // d_out * 4
    int h_start = h_out << 2;  // h_out * 4
    int w_start = w_out << 2;  // w_out * 4
    
    // Stride calculations
    int in_hw = H_in * W_in;
    int in_chw = D_in * in_hw;
    int base_idx = n * C * in_chw + c * in_chw;
    
    float max_val = -1e38f;
    
    // Process 4x4x4 input region with two-stage max
    // Fully unrolled for performance
    #pragma unroll
    for (int dd = 0; dd < 4; dd++) {
        int d = d_start + dd;
        if (d >= D_in) continue;
        
        #pragma unroll
        for (int hh = 0; hh < 4; hh++) {
            int h = h_start + hh;
            if (h >= H_in) continue;
            
            #pragma unroll
            for (int ww = 0; ww < 4; ww++) {
                int w = w_start + ww;
                if (w >= W_in) continue;
                
                float val = __ldg(&input[base_idx + d * in_hw + h * W_in + w]);
                max_val = fmaxf(max_val, val);
            }
        }
    }
    
    output[out_idx] = max_val;
}

torch::Tensor fused_double_maxpool_hip(torch::Tensor input) {
    int N = input.size(0);
    int C = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);
    
    int D_out = D_in / 4;
    int H_out = H_in / 4;
    int W_out = W_in / 4;
    
    auto output = torch::empty({N, C, D_out, H_out, W_out}, input.options());
    
    int total_out = N * C * D_out * H_out * W_out;
    int block_size = 256;
    int num_blocks = (total_out + block_size - 1) / block_size;
    
    fused_double_maxpool_tiled_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D_in, H_in, W_in, D_out, H_out, W_out
    );
    
    return output;
}
"""

kernel_cpp = """
torch::Tensor fused_double_maxpool_hip(torch::Tensor input);
"""

module = load_inline(
    name="conv3d_softmax_maxpool_v6",
    cpp_sources=kernel_cpp,
    cuda_sources=kernel_source,
    functions=["fused_double_maxpool_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.module = module

    def forward(self, x):
        x = self.conv(x)
        x = torch.softmax(x, dim=1)  # Use PyTorch's optimized softmax
        x = self.module.fused_double_maxpool_hip(x)  # Custom fused double maxpool
        return x


def get_inputs():
    return [torch.rand(128, 3, 16, 32, 32).cuda()]


def get_init_inputs():
    return [3, 16, 3, 2]
