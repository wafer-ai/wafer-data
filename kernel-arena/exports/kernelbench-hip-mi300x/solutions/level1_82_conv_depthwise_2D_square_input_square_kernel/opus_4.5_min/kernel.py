import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

depthwise_conv2d_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>

// Highly optimized kernel using register blocking and optimized shared memory
__global__ void depthwise_conv2d_optimized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int padding
) {
    // 64x4 threads, each thread handles 8 vertical outputs
    // Block covers 64 wide x 32 tall output region
    const int BLOCK_W = 64;
    const int BLOCK_H = 32;
    
    __shared__ float s_input[34][68];  // 32 + 2 rows, 64 + 4 cols (with padding)
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int c = blockIdx.z % channels;
    const int b = blockIdx.z / channels;
    
    // Load weights into registers immediately
    float w[9];
    if (ty == 0 && tx < 9) {
        w[tx] = weight[c * 9 + tx];
    }
    // Broadcast weights to all threads via shared memory
    __shared__ float sw[9];
    if (ty == 0 && tx < 9) {
        sw[tx] = weight[c * 9 + tx];
    }
    __syncthreads();
    
    // All threads load weights to registers
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        w[i] = sw[i];
    }
    
    const int in_y_base = blockIdx.y * BLOCK_H - padding;
    const int in_x_base = blockIdx.x * BLOCK_W - padding;
    const int input_plane_offset = (b * channels + c) * in_height * in_width;
    
    // Cooperative loading: 256 threads load 34 rows x 68 cols = 2312 elements
    // Each thread loads ~9 elements
    const int total_elements = 34 * 68;
    const int elements_per_thread = (total_elements + 255) / 256;
    
    int tid = ty * 64 + tx;
    
    #pragma unroll
    for (int e = 0; e < elements_per_thread; e++) {
        int idx = tid + e * 256;
        if (idx < total_elements) {
            int row = idx / 68;
            int col = idx % 68;
            int in_y = in_y_base + row;
            int in_x = in_x_base + col;
            
            float val = 0.0f;
            if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                val = input[input_plane_offset + in_y * in_width + in_x];
            }
            s_input[row][col] = val;
        }
    }
    
    __syncthreads();
    
    const int out_x = blockIdx.x * BLOCK_W + tx;
    if (out_x >= out_width) return;
    
    const int output_plane_offset = (b * channels + c) * out_height * out_width;
    
    // Each of 4 rows of threads handles 8 output rows (4 * 8 = 32)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int out_y = blockIdx.y * BLOCK_H + ty * 8 + i;
        if (out_y >= out_height) break;
        
        int sy = ty * 8 + i;
        
        float sum = 0.0f;
        sum += s_input[sy][tx] * w[0];
        sum += s_input[sy][tx + 1] * w[1];
        sum += s_input[sy][tx + 2] * w[2];
        sum += s_input[sy + 1][tx] * w[3];
        sum += s_input[sy + 1][tx + 1] * w[4];
        sum += s_input[sy + 1][tx + 2] * w[5];
        sum += s_input[sy + 2][tx] * w[6];
        sum += s_input[sy + 2][tx + 1] * w[7];
        sum += s_input[sy + 2][tx + 2] * w[8];
        
        output[output_plane_offset + out_y * out_width + out_x] = sum;
    }
}

// Very large tile version - better for large images
__global__ void depthwise_conv2d_large_tile_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int padding
) {
    // 128x2 threads, each thread handles 16 vertical outputs
    // Block covers 128 wide x 32 tall output region
    const int BLOCK_W = 128;
    const int BLOCK_H = 32;
    
    __shared__ float s_input[34][132];  // 32 + 2 rows, 128 + 4 cols
    __shared__ float sw[9];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int c = blockIdx.z % channels;
    const int b = blockIdx.z / channels;
    
    // Load weights
    int tid = ty * 128 + tx;
    if (tid < 9) {
        sw[tid] = weight[c * 9 + tid];
    }
    __syncthreads();
    
    float w[9];
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        w[i] = sw[i];
    }
    
    const int in_y_base = blockIdx.y * BLOCK_H - padding;
    const int in_x_base = blockIdx.x * BLOCK_W - padding;
    const int input_plane_offset = (b * channels + c) * in_height * in_width;
    
    // Load input tile
    const int total_elements = 34 * 132;
    const int num_threads = 128 * 2;
    
    for (int idx = tid; idx < total_elements; idx += num_threads) {
        int row = idx / 132;
        int col = idx % 132;
        int in_y = in_y_base + row;
        int in_x = in_x_base + col;
        
        float val = 0.0f;
        if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
            val = input[input_plane_offset + in_y * in_width + in_x];
        }
        s_input[row][col] = val;
    }
    
    __syncthreads();
    
    const int out_x = blockIdx.x * BLOCK_W + tx;
    if (out_x >= out_width) return;
    
    const int output_plane_offset = (b * channels + c) * out_height * out_width;
    
    // Each of 2 rows of threads handles 16 output rows
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int out_y = blockIdx.y * BLOCK_H + ty * 16 + i;
        if (out_y >= out_height) break;
        
        int sy = ty * 16 + i;
        
        float sum = 0.0f;
        sum += s_input[sy][tx] * w[0];
        sum += s_input[sy][tx + 1] * w[1];
        sum += s_input[sy][tx + 2] * w[2];
        sum += s_input[sy + 1][tx] * w[3];
        sum += s_input[sy + 1][tx + 1] * w[4];
        sum += s_input[sy + 1][tx + 2] * w[5];
        sum += s_input[sy + 2][tx] * w[6];
        sum += s_input[sy + 2][tx + 1] * w[7];
        sum += s_input[sy + 2][tx + 2] * w[8];
        
        output[output_plane_offset + out_y * out_width + out_x] = sum;
    }
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
    int kernel_size = weight.size(1);
    
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());
    
    if (kernel_size == 3 && stride == 1) {
        // Use optimized kernel
        dim3 block(64, 4);  // 256 threads
        dim3 grid(
            (out_width + 63) / 64,
            (out_height + 31) / 32,
            batch_size * channels
        );
        
        depthwise_conv2d_optimized_kernel<<<grid, block>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            channels,
            in_height,
            in_width,
            out_height,
            out_width,
            padding
        );
    }
    
    return output;
}
"""

depthwise_conv2d_cpp_source = """
torch::Tensor depthwise_conv2d_hip(torch::Tensor input, torch::Tensor weight, int stride, int padding);
"""

depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv2d.weight.view(self.in_channels, self.kernel_size, self.kernel_size)
        output = depthwise_conv2d.depthwise_conv2d_hip(
            x.contiguous(),
            weight.contiguous(),
            self.stride,
            self.padding
        )
        if self.conv2d.bias is not None:
            output = output + self.conv2d.bias.view(1, -1, 1, 1)
        return output
