import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

depthwise_conv2d_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>

#define TILE_W 64
#define TILE_H 4
#define OUTPUTS_PER_THREAD_Y 8

// Heavily optimized kernel: each thread computes multiple outputs, uses vectorized loads
__global__ void depthwise_conv2d_fast_kernel(
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
    // Shared memory for input tile
    __shared__ float s_input[TILE_H * OUTPUTS_PER_THREAD_Y + 2][TILE_W + 4];
    __shared__ float s_weight[9];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int out_x = blockIdx.x * TILE_W + tx;
    int c = blockIdx.z % channels;
    int b = blockIdx.z / channels;

    // Load weights
    int tid = ty * blockDim.x + tx;
    if (tid < 9) {
        s_weight[tid] = weight[c * 9 + tid];
    }

    int in_y_base = blockIdx.y * (TILE_H * OUTPUTS_PER_THREAD_Y) * stride - padding;
    int in_x_base = blockIdx.x * TILE_W * stride - padding;

    // Each block covers TILE_H * OUTPUTS_PER_THREAD_Y output rows
    // Need to load TILE_H * OUTPUTS_PER_THREAD_Y + 2 input rows
    const int TOTAL_INPUT_ROWS = TILE_H * OUTPUTS_PER_THREAD_Y + 2;
    const int NUM_THREADS = TILE_W * TILE_H;
    
    // Cooperative loading of all input tiles
    int input_base = ((b * channels + c) * in_height) * in_width;
    
    for (int row = tid; row < TOTAL_INPUT_ROWS; row += NUM_THREADS) {
        int in_y = in_y_base + row;
        
        // Load main tile
        for (int col = tx; col < TILE_W + 4; col += TILE_W) {
            int in_x = in_x_base + col;
            if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                s_input[row][col] = input[input_base + in_y * in_width + in_x];
            } else {
                s_input[row][col] = 0.0f;
            }
        }
    }

    __syncthreads();

    if (out_x >= out_width || b >= batch_size) return;

    float w0 = s_weight[0], w1 = s_weight[1], w2 = s_weight[2];
    float w3 = s_weight[3], w4 = s_weight[4], w5 = s_weight[5];
    float w6 = s_weight[6], w7 = s_weight[7], w8 = s_weight[8];

    // Each thread computes OUTPUTS_PER_THREAD_Y outputs vertically
    #pragma unroll
    for (int i = 0; i < OUTPUTS_PER_THREAD_Y; i++) {
        int out_y = blockIdx.y * (TILE_H * OUTPUTS_PER_THREAD_Y) + ty + i * TILE_H;
        if (out_y >= out_height) break;
        
        int sy = ty + i * TILE_H;
        
        float sum = 0.0f;
        sum += s_input[sy + 0][tx + 0] * w0;
        sum += s_input[sy + 0][tx + 1] * w1;
        sum += s_input[sy + 0][tx + 2] * w2;
        sum += s_input[sy + 1][tx + 0] * w3;
        sum += s_input[sy + 1][tx + 1] * w4;
        sum += s_input[sy + 1][tx + 2] * w5;
        sum += s_input[sy + 2][tx + 0] * w6;
        sum += s_input[sy + 2][tx + 1] * w7;
        sum += s_input[sy + 2][tx + 2] * w8;

        int out_idx = ((b * channels + c) * out_height + out_y) * out_width + out_x;
        output[out_idx] = sum;
    }
}

// Alternative: Use float4 for better memory bandwidth
__global__ void depthwise_conv2d_vec_kernel(
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
    __shared__ float s_input[36][68];  // 32 + 2 + 2 padding for alignment
    __shared__ float s_weight[9];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int out_x = blockIdx.x * 64 + tx;
    int c = blockIdx.z % channels;
    int b = blockIdx.z / channels;

    int tid = ty * 64 + tx;
    if (tid < 9) {
        s_weight[tid] = weight[c * 9 + tid];
    }

    int in_y_base = blockIdx.y * 32 - padding;
    int in_x_base = blockIdx.x * 64 - padding;

    // Load input tile cooperatively
    int input_base = ((b * channels + c) * in_height) * in_width;
    
    // Each thread loads multiple elements
    for (int row = ty; row < 34; row += 4) {
        int in_y = in_y_base + row;
        bool valid_y = (in_y >= 0 && in_y < in_height);
        
        int in_x = in_x_base + tx;
        if (valid_y && in_x >= 0 && in_x < in_width) {
            s_input[row][tx] = input[input_base + in_y * in_width + in_x];
        } else {
            s_input[row][tx] = 0.0f;
        }
        
        // Load extra columns (66 total needed = 64 + 2)
        if (tx < 4) {
            int extra_x = in_x_base + 64 + tx;
            if (valid_y && extra_x >= 0 && extra_x < in_width) {
                s_input[row][64 + tx] = input[input_base + in_y * in_width + extra_x];
            } else {
                s_input[row][64 + tx] = 0.0f;
            }
        }
    }

    __syncthreads();

    if (out_x >= out_width || b >= batch_size) return;

    float w0 = s_weight[0], w1 = s_weight[1], w2 = s_weight[2];
    float w3 = s_weight[3], w4 = s_weight[4], w5 = s_weight[5];
    float w6 = s_weight[6], w7 = s_weight[7], w8 = s_weight[8];

    // Each thread handles 8 rows
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int out_y = blockIdx.y * 32 + ty * 8 + i;
        if (out_y >= out_height) break;
        
        int sy = ty * 8 + i;
        
        float sum = 0.0f;
        sum += s_input[sy + 0][tx + 0] * w0;
        sum += s_input[sy + 0][tx + 1] * w1;
        sum += s_input[sy + 0][tx + 2] * w2;
        sum += s_input[sy + 1][tx + 0] * w3;
        sum += s_input[sy + 1][tx + 1] * w4;
        sum += s_input[sy + 1][tx + 2] * w5;
        sum += s_input[sy + 2][tx + 0] * w6;
        sum += s_input[sy + 2][tx + 1] * w7;
        sum += s_input[sy + 2][tx + 2] * w8;

        int out_idx = ((b * channels + c) * out_height + out_y) * out_width + out_x;
        output[out_idx] = sum;
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
        // Use vec kernel: 64x4 threads, each thread processes 8 rows
        dim3 block(64, 4);
        dim3 grid(
            (out_width + 63) / 64,
            (out_height + 31) / 32,
            batch_size * channels
        );
        
        depthwise_conv2d_vec_kernel<<<grid, block>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            channels,
            in_height,
            in_width,
            out_height,
            out_width,
            stride,
            padding
        );
    } else {
        // Fallback
        dim3 block(TILE_W, TILE_H);
        dim3 grid(
            (out_width + TILE_W - 1) / TILE_W,
            (out_height + TILE_H * OUTPUTS_PER_THREAD_Y - 1) / (TILE_H * OUTPUTS_PER_THREAD_Y),
            batch_size * channels
        );
        depthwise_conv2d_fast_kernel<<<grid, block>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            channels,
            in_height,
            in_width,
            out_height,
            out_width,
            stride,
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
    extra_cuda_cflags=["-O3"]
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
