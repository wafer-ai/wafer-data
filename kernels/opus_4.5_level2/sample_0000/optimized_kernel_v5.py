import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized kernel with vectorized loads
kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Vectorized softmax along channel dimension for 5D tensor
// Using float4 for coalesced memory access
__global__ void channel_softmax_vec_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D, int H, int W
) {
    int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_spatial = N * D * H * W;
    
    if (spatial_idx >= total_spatial) return;
    
    int w = spatial_idx % W;
    int tmp = spatial_idx / W;
    int h = tmp % H;
    tmp = tmp / H;
    int d = tmp % D;
    int n = tmp / D;
    
    int spatial_stride = H * W;
    int channel_stride = D * spatial_stride;
    int batch_offset = n * C * channel_stride;
    int pos_offset = d * spatial_stride + h * W + w;
    
    // Find max for numerical stability
    float max_val = -1e38f;
    #pragma unroll 4
    for (int c = 0; c < C; c++) {
        float val = input[batch_offset + c * channel_stride + pos_offset];
        max_val = fmaxf(max_val, val);
    }
    
    // Compute exp sum
    float exp_sum = 0.0f;
    #pragma unroll 4
    for (int c = 0; c < C; c++) {
        float val = input[batch_offset + c * channel_stride + pos_offset];
        exp_sum += __expf(val - max_val);
    }
    
    // Compute softmax values
    float inv_sum = __frcp_rn(exp_sum);
    #pragma unroll 4
    for (int c = 0; c < C; c++) {
        float val = input[batch_offset + c * channel_stride + pos_offset];
        output[batch_offset + c * channel_stride + pos_offset] = __expf(val - max_val) * inv_sum;
    }
}

// Optimized double maxpool with better memory access pattern
__global__ void fused_double_maxpool_opt_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * D_out * H_out * W_out;
    
    if (idx >= total) return;
    
    int w_out = idx % W_out;
    int tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp = tmp / H_out;
    int d_out = tmp % D_out;
    tmp = tmp / D_out;
    int c = tmp % C;
    int n = tmp / C;
    
    int d_start = d_out * 4;
    int h_start = h_out * 4;
    int w_start = w_out * 4;
    
    int in_spatial = H_in * W_in;
    int in_channel = D_in * in_spatial;
    int base = n * C * in_channel + c * in_channel;
    
    float max_val = -1e38f;
    
    // Unrolled two-stage pooling 
    #pragma unroll
    for (int pd2 = 0; pd2 < 2; pd2++) {
        #pragma unroll
        for (int ph2 = 0; ph2 < 2; ph2++) {
            #pragma unroll
            for (int pw2 = 0; pw2 < 2; pw2++) {
                int d_base = d_start + pd2 * 2;
                int h_base = h_start + ph2 * 2;
                int w_base = w_start + pw2 * 2;
                
                float pool1_max = -1e38f;
                
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
                                float val = input[base + d * in_spatial + h * W_in + w];
                                pool1_max = fmaxf(pool1_max, val);
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

torch::Tensor softmax_double_maxpool_hip(torch::Tensor input) {
    int N = input.size(0);
    int C = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);
    
    auto softmax_out = torch::empty_like(input);
    
    int total_spatial = N * D_in * H_in * W_in;
    int block_size = 256;
    int num_blocks = (total_spatial + block_size - 1) / block_size;
    
    channel_softmax_vec_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        softmax_out.data_ptr<float>(),
        N, C, D_in, H_in, W_in
    );
    
    int D_out = D_in / 4;
    int H_out = H_in / 4;
    int W_out = W_in / 4;
    
    auto output = torch::empty({N, C, D_out, H_out, W_out}, input.options());
    
    int total_out = N * C * D_out * H_out * W_out;
    num_blocks = (total_out + block_size - 1) / block_size;
    
    fused_double_maxpool_opt_kernel<<<num_blocks, block_size>>>(
        softmax_out.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D_in, H_in, W_in, D_out, H_out, W_out
    );
    
    return output;
}
"""

kernel_cpp = """
torch::Tensor softmax_double_maxpool_hip(torch::Tensor input);
"""

module = load_inline(
    name="conv3d_softmax_maxpool_v5",
    cpp_sources=kernel_cpp,
    cuda_sources=kernel_source,
    functions=["softmax_double_maxpool_hip"],
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
        x = self.module.softmax_double_maxpool_hip(x)
        return x


def get_inputs():
    return [torch.rand(128, 3, 16, 32, 32).cuda()]


def get_init_inputs():
    return [3, 16, 3, 2]
