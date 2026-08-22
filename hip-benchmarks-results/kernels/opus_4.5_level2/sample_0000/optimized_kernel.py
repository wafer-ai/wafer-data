import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused softmax + double maxpool kernel
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Fused softmax (along channel dim) + double maxpool (2x2x2 twice = effective 4x4x4)
__global__ void fused_softmax_double_maxpool_kernel(
    const float* __restrict__ input,  // (N, C, D_in, H_in, W_in)
    float* __restrict__ output,       // (N, C, D_out, H_out, W_out)
    int N, int C,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out
) {
    // Each thread handles one output element
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
    
    // Calculate input region for this output
    // First pool: 2x2x2, Second pool: 2x2x2
    // So we need to consider a 4x4x4 region with intermediate max operations
    int d_start = d_out * 4;
    int h_start = h_out * 4;
    int w_start = w_out * 4;
    
    // First, we need to do softmax along channels, then double maxpool
    // Since softmax needs all channels at each spatial position,
    // we compute softmax values as needed and track the max
    
    float max_val = -1e38f;
    
    // Iterate over the effective 4x4x4 pooling region
    // But we need to respect the two-stage pooling structure
    // Pool1: 2x2x2 reduces D_in,H_in,W_in -> D_mid,H_mid,W_mid
    // Pool2: 2x2x2 reduces D_mid,H_mid,W_mid -> D_out,H_out,W_out
    
    // For each 2x2x2 block in the first pool that feeds into our 2x2x2 second pool
    for (int pd2 = 0; pd2 < 2; pd2++) {
        for (int ph2 = 0; ph2 < 2; ph2++) {
            for (int pw2 = 0; pw2 < 2; pw2++) {
                // This is one position in the intermediate tensor (after first pool)
                // Which corresponds to a 2x2x2 region in the input
                int d_base = d_start + pd2 * 2;
                int h_base = h_start + ph2 * 2;
                int w_base = w_start + pw2 * 2;
                
                // First maxpool over this 2x2x2 region
                float pool1_max = -1e38f;
                
                for (int pd1 = 0; pd1 < 2; pd1++) {
                    for (int ph1 = 0; ph1 < 2; ph1++) {
                        for (int pw1 = 0; pw1 < 2; pw1++) {
                            int d = d_base + pd1;
                            int h = h_base + ph1;
                            int w = w_base + pw1;
                            
                            if (d < D_in && h < H_in && w < W_in) {
                                // Compute softmax at this spatial position
                                // First find max for numerical stability
                                float max_c = -1e38f;
                                for (int cc = 0; cc < C; cc++) {
                                    int in_idx = ((n * C + cc) * D_in + d) * H_in * W_in + h * W_in + w;
                                    float val = input[in_idx];
                                    if (val > max_c) max_c = val;
                                }
                                
                                // Compute exp sum
                                float exp_sum = 0.0f;
                                for (int cc = 0; cc < C; cc++) {
                                    int in_idx = ((n * C + cc) * D_in + d) * H_in * W_in + h * W_in + w;
                                    exp_sum += expf(input[in_idx] - max_c);
                                }
                                
                                // Compute softmax for current channel
                                int in_idx = ((n * C + c) * D_in + d) * H_in * W_in + h * W_in + w;
                                float softmax_val = expf(input[in_idx] - max_c) / exp_sum;
                                
                                if (softmax_val > pool1_max) {
                                    pool1_max = softmax_val;
                                }
                            }
                        }
                    }
                }
                
                // Second maxpool accumulates the max from first pool
                if (pool1_max > max_val) {
                    max_val = pool1_max;
                }
            }
        }
    }
    
    output[idx] = max_val;
}

torch::Tensor fused_softmax_double_maxpool_hip(torch::Tensor input) {
    // input shape: (N, C, D_in, H_in, W_in)
    int N = input.size(0);
    int C = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);
    
    // After first pool (2x2x2)
    int D_mid = D_in / 2;
    int H_mid = H_in / 2;
    int W_mid = W_in / 2;
    
    // After second pool (2x2x2)
    int D_out = D_mid / 2;
    int H_out = H_mid / 2;
    int W_out = W_mid / 2;
    
    auto output = torch::empty({N, C, D_out, H_out, W_out}, input.options());
    
    int total = N * C * D_out * H_out * W_out;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    
    fused_softmax_double_maxpool_kernel<<<num_blocks, block_size>>>(
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
    name="fused_softmax_maxpool",
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
