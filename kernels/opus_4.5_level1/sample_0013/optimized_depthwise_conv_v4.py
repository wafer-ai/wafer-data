import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

depthwise_conv_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Highly optimized 3x3 depthwise convolution
// Uses larger tiles and better memory access patterns
__global__ void depthwise_conv2d_3x3_fast(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int stride,
    int padding
) {
    // Tile dimensions: each block processes 64x16 output elements
    // 64 threads per row (each handles 1 element), 16 rows
    constexpr int TILE_W = 64;
    constexpr int TILE_H = 16;
    constexpr int smem_w = TILE_W + 2;  // For 3x3 kernel halo
    constexpr int smem_h = TILE_H + 2;
    
    int bc = blockIdx.z;
    int batch = bc / channels;
    int channel = bc % channels;
    
    int tile_x = blockIdx.x * TILE_W;
    int tile_y = blockIdx.y * TILE_H;
    
    int tx = threadIdx.x;  // 0-63
    int ty = threadIdx.y;  // 0-15
    
    // Load weights into registers
    float w[9];
    const float* weight_ptr = weight + channel * 9;
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        w[i] = weight_ptr[i];
    }
    
    const float* input_base = input + (batch * channels + channel) * in_height * in_width;
    float* output_base = output + (batch * channels + channel) * out_height * out_width;
    
    // Shared memory for input tile
    __shared__ float smem[smem_h][smem_w + 1];  // +1 to avoid bank conflicts
    
    int in_tile_x = tile_x - padding;
    int in_tile_y = tile_y - padding;
    
    // Cooperative loading using all 1024 threads
    int thread_id = ty * TILE_W + tx;
    int total_threads = TILE_W * TILE_H;  // 1024
    int total_elements = smem_h * smem_w;  // 18 * 66 = 1188
    
    for (int idx = thread_id; idx < total_elements; idx += total_threads) {
        int sy = idx / smem_w;
        int sx = idx % smem_w;
        
        int in_y = in_tile_y + sy;
        int in_x = in_tile_x + sx;
        
        float val = 0.0f;
        if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
            val = input_base[in_y * in_width + in_x];
        }
        smem[sy][sx] = val;
    }
    
    __syncthreads();
    
    // Compute output
    int out_x = tile_x + tx;
    int out_y = tile_y + ty;
    
    if (out_x < out_width && out_y < out_height) {
        // Local indices in shared memory
        int ly = ty;
        int lx = tx;
        
        float sum = smem[ly][lx] * w[0]
                  + smem[ly][lx+1] * w[1]
                  + smem[ly][lx+2] * w[2]
                  + smem[ly+1][lx] * w[3]
                  + smem[ly+1][lx+1] * w[4]
                  + smem[ly+1][lx+2] * w[5]
                  + smem[ly+2][lx] * w[6]
                  + smem[ly+2][lx+1] * w[7]
                  + smem[ly+2][lx+2] * w[8];
        
        output_base[out_y * out_width + out_x] = sum;
    }
}

// Alternative: Process 2 output elements per thread for even better throughput
__global__ void depthwise_conv2d_3x3_x2(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int stride,
    int padding
) {
    constexpr int TILE_W = 128;
    constexpr int TILE_H = 8;
    constexpr int smem_w = TILE_W + 2;
    constexpr int smem_h = TILE_H + 2;
    
    int bc = blockIdx.z;
    int batch = bc / channels;
    int channel = bc % channels;
    
    int tile_x = blockIdx.x * TILE_W;
    int tile_y = blockIdx.y * TILE_H;
    
    int tx = threadIdx.x;  // 0-63
    int ty = threadIdx.y;  // 0-7
    
    // Load weights
    float w[9];
    const float* weight_ptr = weight + channel * 9;
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        w[i] = weight_ptr[i];
    }
    
    const float* input_base = input + (batch * channels + channel) * in_height * in_width;
    float* output_base = output + (batch * channels + channel) * out_height * out_width;
    
    __shared__ float smem[smem_h][smem_w + 1];
    
    int in_tile_x = tile_x - padding;
    int in_tile_y = tile_y - padding;
    
    // Cooperative loading
    int thread_id = ty * 64 + tx;
    int total_threads = 64 * 8;  // 512
    int total_elements = smem_h * smem_w;  // 10 * 130 = 1300
    
    for (int idx = thread_id; idx < total_elements; idx += total_threads) {
        int sy = idx / smem_w;
        int sx = idx % smem_w;
        
        int in_y = in_tile_y + sy;
        int in_x = in_tile_x + sx;
        
        float val = 0.0f;
        if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
            val = input_base[in_y * in_width + in_x];
        }
        smem[sy][sx] = val;
    }
    
    __syncthreads();
    
    int out_y = tile_y + ty;
    
    if (out_y < out_height) {
        int ly = ty;
        
        // Each thread processes 2 consecutive output pixels
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int out_x = tile_x + tx * 2 + i;
            if (out_x < out_width) {
                int lx = tx * 2 + i;
                
                float sum = smem[ly][lx] * w[0]
                          + smem[ly][lx+1] * w[1]
                          + smem[ly][lx+2] * w[2]
                          + smem[ly+1][lx] * w[3]
                          + smem[ly+1][lx+1] * w[4]
                          + smem[ly+1][lx+2] * w[5]
                          + smem[ly+2][lx] * w[6]
                          + smem[ly+2][lx+1] * w[7]
                          + smem[ly+2][lx+2] * w[8];
                
                output_base[out_y * out_width + out_x] = sum;
            }
        }
    }
}

// Generic kernel for other kernel sizes
__global__ void depthwise_conv2d_generic(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int bc = blockIdx.z;
    int batch = bc / channels;
    int channel = bc % channels;
    
    if (out_x >= out_width || out_y >= out_height) return;
    
    float sum = 0.0f;
    
    int in_y_start = out_y * stride - padding;
    int in_x_start = out_x * stride - padding;
    
    const float* weight_ptr = weight + channel * kernel_size * kernel_size;
    const float* input_ptr = input + (batch * channels + channel) * in_height * in_width;
    
    for (int ky = 0; ky < kernel_size; ++ky) {
        int in_y = in_y_start + ky;
        if (in_y >= 0 && in_y < in_height) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int in_x = in_x_start + kx;
                if (in_x >= 0 && in_x < in_width) {
                    sum += input_ptr[in_y * in_width + in_x] * weight_ptr[ky * kernel_size + kx];
                }
            }
        }
    }
    
    output[(batch * channels + channel) * out_height * out_width + out_y * out_width + out_x] = sum;
}

torch::Tensor depthwise_conv2d_hip(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int kernel_size = weight.size(2);
    
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());
    
    if (kernel_size == 3 && stride == 1) {
        // Use the x2 kernel for better throughput
        constexpr int TILE_W = 128;
        constexpr int TILE_H = 8;
        dim3 block(64, 8);  // 512 threads
        dim3 grid(
            (out_width + TILE_W - 1) / TILE_W,
            (out_height + TILE_H - 1) / TILE_H,
            batch_size * channels
        );
        depthwise_conv2d_3x3_x2<<<grid, block>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, channels,
            in_height, in_width,
            out_height, out_width,
            stride, padding
        );
    } else {
        dim3 block(16, 16);
        dim3 grid(
            (out_width + 15) / 16,
            (out_height + 15) / 16,
            batch_size * channels
        );
        depthwise_conv2d_generic<<<grid, block>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, channels,
            in_height, in_width,
            out_height, out_width,
            kernel_size, stride, padding
        );
    }
    
    return output;
}
"""

cpp_source = """
torch::Tensor depthwise_conv2d_hip(torch::Tensor input, torch::Tensor weight, int stride, int padding);
"""

depthwise_conv_module = load_inline(
    name="depthwise_conv2d",
    cpp_sources=cpp_source,
    cuda_sources=depthwise_conv_source,
    functions=["depthwise_conv2d_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution using custom HIP kernel.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_flag = bias
        
        # Initialize weights the same way PyTorch does
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=2.236)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape weight from (C, 1, K, K) to (C, K, K) for our kernel
        weight_reshaped = self.weight.squeeze(1)
        
        output = depthwise_conv_module.depthwise_conv2d_hip(
            x.contiguous(),
            weight_reshaped.contiguous(),
            self.stride,
            self.padding
        )
        
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        
        return output
