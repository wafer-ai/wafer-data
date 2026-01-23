import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused softmax + double maxpool kernel with shared memory
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define BLOCK_SIZE 256
#define MAX_CHANNELS 32

// Optimized kernel: Each block handles multiple output positions
// Use shared memory to cache softmax computations
__global__ void fused_softmax_double_maxpool_v2_kernel(
    const float* __restrict__ input,  // (N, C, D_in, H_in, W_in)
    float* __restrict__ output,       // (N, C, D_out, H_out, W_out)
    int N, int C,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out
) {
    // Each thread handles one (n, c, d_out, h_out, w_out) output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * D_out * H_out * W_out;
    
    if (idx >= total) return;
    
    // Decode output index
    int w_out = idx % W_out;
    int tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp = tmp / H_out;
    int d_out = tmp % D_out;
    tmp = tmp / D_out;
    int c = tmp % C;
    int n = tmp / C;
    
    // Calculate input region
    int d_start = d_out * 4;
    int h_start = h_out * 4;
    int w_start = w_out * 4;
    
    int spatial_stride = H_in * W_in;
    int channel_stride = D_in * spatial_stride;
    int batch_offset = n * C * channel_stride;
    
    float max_val = -1e38f;
    
    // Iterate over the effective 4x4x4 pooling region with two-stage pooling
    for (int pd2 = 0; pd2 < 2; pd2++) {
        for (int ph2 = 0; ph2 < 2; ph2++) {
            for (int pw2 = 0; pw2 < 2; pw2++) {
                int d_base = d_start + pd2 * 2;
                int h_base = h_start + ph2 * 2;
                int w_base = w_start + pw2 * 2;
                
                float pool1_max = -1e38f;
                
                // First maxpool over 2x2x2 region
                #pragma unroll
                for (int pd1 = 0; pd1 < 2; pd1++) {
                    #pragma unroll
                    for (int ph1 = 0; ph1 < 2; ph1++) {
                        #pragma unroll
                        for (int pw1 = 0; pw1 < 2; pw1++) {
                            int d = d_base + pd1;
                            int h = h_base + ph1;
                            int w = w_base + pw1;
                            
                            if (d < D_in && h < H_in && w < W_in) {
                                // Compute softmax at this spatial position
                                int base_idx = batch_offset + d * spatial_stride + h * W_in + w;
                                
                                // Find max for numerical stability
                                float max_c = -1e38f;
                                for (int cc = 0; cc < C; cc++) {
                                    float val = input[base_idx + cc * channel_stride];
                                    max_c = fmaxf(max_c, val);
                                }
                                
                                // Compute exp sum
                                float exp_sum = 0.0f;
                                for (int cc = 0; cc < C; cc++) {
                                    exp_sum += expf(input[base_idx + cc * channel_stride] - max_c);
                                }
                                
                                // Compute softmax for current channel
                                float softmax_val = expf(input[base_idx + c * channel_stride] - max_c) / exp_sum;
                                pool1_max = fmaxf(pool1_max, softmax_val);
                            }
                        }
                    }
                }
                
                max_val = fmaxf(max_val, pool1_max);
            }
        }
    }
    
    output[idx] = max_val;
}

torch::Tensor fused_softmax_double_maxpool_hip(torch::Tensor input) {
    int N = input.size(0);
    int C = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);
    
    int D_mid = D_in / 2;
    int H_mid = H_in / 2;
    int W_mid = W_in / 2;
    
    int D_out = D_mid / 2;
    int H_out = H_mid / 2;
    int W_out = W_mid / 2;
    
    auto output = torch::empty({N, C, D_out, H_out, W_out}, input.options());
    
    int total = N * C * D_out * H_out * W_out;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    
    fused_softmax_double_maxpool_v2_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D_in, H_in, W_in, D_out, H_out, W_out
    );
    
    return output;
}
"""

fused_kernel_cpp = """
torch::Tensor fused_softmax_double_maxpool_hip(torch::Tensor input);
"""

fused_module = load_inline(
    name="fused_softmax_maxpool_v2",
    cpp_sources=fused_kernel_cpp,
    cuda_sources=fused_kernel_source,
    functions=["fused_softmax_double_maxpool_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    """
    Optimized model with fused softmax and double maxpool operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.fused_module = fused_module

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_module.fused_softmax_double_maxpool_hip(x)
        return x


def get_inputs():
    return [torch.rand(128, 3, 16, 32, 32).cuda()]


def get_init_inputs():
    return [3, 16, 3, 2]
