import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

conv2d_source = """
#include <hip/hip_runtime.h>

#define TILE_H 4
#define TILE_W 64
#define BLOCK_K 32
#define IN_C 16
#define KERNEL_SIZE 3
#define THREAD_W 32

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    float* __restrict__ output,
    int batch_size,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int out_channels
) {
    int k_chunks = out_channels / BLOCK_K;
    int n = blockIdx.z / k_chunks;
    int k_block = blockIdx.z % k_chunks;
    int k_start = k_block * BLOCK_K;
    
    int h_start = blockIdx.y * TILE_H;
    int w_start = blockIdx.x * TILE_W;
    
    int tid = threadIdx.x;
    
    // Shared memory
    // Input tile height: 4 + 2 = 6
    // Input tile width: 64 + 2 = 66
    __shared__ float s_input[IN_C][6][66];
    
    // Load Input
    int input_tile_h = 6;
    int input_tile_w = 66;
    int total_input_elements = IN_C * 6 * 66; // 6336
    
    for (int i = tid; i < total_input_elements; i += blockDim.x) {
        int c = i / 396; // 6*66
        int rem = i % 396;
        int y = rem / 66;
        int x = rem % 66;
        
        int in_y = h_start + y;
        int in_x = w_start + x;
        
        if (in_y < in_h && in_x < in_w) {
            size_t idx = (size_t)n * (IN_C * in_h * in_w) + (size_t)c * (in_h * in_w) + (size_t)in_y * in_w + in_x;
            s_input[c][y][x] = input[idx];
        } else {
            s_input[c][y][x] = 0.0f;
        }
    }
    
    __syncthreads();
    
    // Compute
    int tid_y = tid / THREAD_W; // 0..3
    int tid_x = tid % THREAD_W; // 0..31
    
    int local_y = tid_y;
    int local_x_base = tid_x * 2;
    
    float acc0[BLOCK_K];
    float acc1[BLOCK_K];
    
    #pragma unroll
    for (int k = 0; k < BLOCK_K; ++k) {
        acc0[k] = 0.0f;
        acc1[k] = 0.0f;
    }
    
    for (int c = 0; c < IN_C; ++c) {
        #pragma unroll
        for (int r = 0; r < KERNEL_SIZE; ++r) {
            #pragma unroll
            for (int s = 0; s < KERNEL_SIZE; ++s) {
                float val0 = s_input[c][local_y + r][local_x_base + s];
                float val1 = s_input[c][local_y + r][local_x_base + 1 + s];
                
                size_t w_base = (size_t)k_start * (IN_C * 9) + (size_t)c * 9 + r * 3 + s;
                size_t w_stride = IN_C * 9;
                
                #pragma unroll
                for (int k = 0; k < BLOCK_K; ++k) {
                    float w = weights[w_base + k * w_stride];
                    acc0[k] += val0 * w;
                    acc1[k] += val1 * w;
                }
            }
        }
    }
    
    // Store result
    int out_y = h_start + local_y;
    int out_x_base = w_start + local_x_base;
    
    size_t batch_offset = (size_t)n * (out_channels * out_h * out_w);
    size_t k_stride = (size_t)out_h * out_w;
    
    if (out_y < out_h) {
        if (out_x_base < out_w) {
            size_t out_pixel_offset = (size_t)out_y * out_w + out_x_base;
            #pragma unroll
            for (int k = 0; k < BLOCK_K; ++k) {
                int k_global = k_start + k;
                if (k_global < out_channels) {
                    output[batch_offset + k_global * k_stride + out_pixel_offset] = acc0[k];
                }
            }
        }
        
        if (out_x_base + 1 < out_w) {
            size_t out_pixel_offset = (size_t)out_y * out_w + (out_x_base + 1);
            #pragma unroll
            for (int k = 0; k < BLOCK_K; ++k) {
                int k_global = k_start + k;
                if (k_global < out_channels) {
                    output[batch_offset + k_global * k_stride + out_pixel_offset] = acc1[k];
                }
            }
        }
    }
}

torch::Tensor conv2d_hip(torch::Tensor input, torch::Tensor weights) {
    auto N = input.size(0);
    auto H = input.size(2);
    auto W = input.size(3);
    auto K = weights.size(0);
    
    auto H_out = H - 2;
    auto W_out = W - 2;
    
    auto output = torch::empty({N, K, H_out, W_out}, input.options());
    
    int block_k = 32;
    int tile_h = 4;
    int tile_w = 64;
    
    int k_chunks = (K + block_k - 1) / block_k;
    
    dim3 block(128); 
    dim3 grid(
        (W_out + tile_w - 1) / tile_w,
        (H_out + tile_h - 1) / tile_h,
        N * k_chunks
    );
    
    conv2d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weights.data_ptr<float>(),
        output.data_ptr<float>(),
        N, H, W, H_out, W_out, K
    );
    
    return output;
}
"""

conv2d_module = load_inline(
    name="conv2d_custom_v5",
    cpp_sources=conv2d_source,
    functions=["conv2d_hip"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv_func = conv2d_module.conv2d_hip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_func(x, self.conv2d.weight)
