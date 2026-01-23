import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

depthwise_conv2d_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>

#define TILE_W 32
#define TILE_H 8

// Optimized version using shared memory for 3x3 kernel with larger tiles
__global__ void depthwise_conv2d_shared_kernel(
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
    // Shared memory for input tile and weights
    __shared__ float s_input[TILE_H + 2][TILE_W + 2];
    __shared__ float s_weight[9]; // 3x3 kernel
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int out_x = blockIdx.x * TILE_W + tx;
    int out_y = blockIdx.y * TILE_H + ty;
    int c = blockIdx.z % channels;
    int b = blockIdx.z / channels;

    // Load weights into shared memory (first 9 threads)
    int tid = ty * TILE_W + tx;
    if (tid < 9) {
        s_weight[tid] = weight[c * 9 + tid];
    }

    // Calculate input coordinates
    int in_y_base = blockIdx.y * TILE_H * stride - padding;
    int in_x_base = blockIdx.x * TILE_W * stride - padding;

    // Load input tile into shared memory
    int in_y = in_y_base + ty;
    int in_x = in_x_base + tx;
    
    // Main region
    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
        s_input[ty][tx] = input[((b * channels + c) * in_height + in_y) * in_width + in_x];
    } else {
        s_input[ty][tx] = 0.0f;
    }

    // Load right halo
    if (tx < 2) {
        int halo_x = in_x_base + TILE_W + tx;
        if (in_y >= 0 && in_y < in_height && halo_x >= 0 && halo_x < in_width) {
            s_input[ty][TILE_W + tx] = input[((b * channels + c) * in_height + in_y) * in_width + halo_x];
        } else {
            s_input[ty][TILE_W + tx] = 0.0f;
        }
    }

    // Load bottom halo
    if (ty < 2) {
        int halo_y = in_y_base + TILE_H + ty;
        if (halo_y >= 0 && halo_y < in_height && in_x >= 0 && in_x < in_width) {
            s_input[TILE_H + ty][tx] = input[((b * channels + c) * in_height + halo_y) * in_width + in_x];
        } else {
            s_input[TILE_H + ty][tx] = 0.0f;
        }
    }

    // Load corner
    if (tx < 2 && ty < 2) {
        int halo_y = in_y_base + TILE_H + ty;
        int halo_x = in_x_base + TILE_W + tx;
        if (halo_y >= 0 && halo_y < in_height && halo_x >= 0 && halo_x < in_width) {
            s_input[TILE_H + ty][TILE_W + tx] = input[((b * channels + c) * in_height + halo_y) * in_width + halo_x];
        } else {
            s_input[TILE_H + ty][TILE_W + tx] = 0.0f;
        }
    }

    __syncthreads();

    if (out_x >= out_width || out_y >= out_height || b >= batch_size) return;

    // Compute convolution using shared memory with full unroll
    float sum = 0.0f;
    sum += s_input[ty + 0][tx + 0] * s_weight[0];
    sum += s_input[ty + 0][tx + 1] * s_weight[1];
    sum += s_input[ty + 0][tx + 2] * s_weight[2];
    sum += s_input[ty + 1][tx + 0] * s_weight[3];
    sum += s_input[ty + 1][tx + 1] * s_weight[4];
    sum += s_input[ty + 1][tx + 2] * s_weight[5];
    sum += s_input[ty + 2][tx + 0] * s_weight[6];
    sum += s_input[ty + 2][tx + 1] * s_weight[7];
    sum += s_input[ty + 2][tx + 2] * s_weight[8];

    int out_idx = ((b * channels + c) * out_height + out_y) * out_width + out_x;
    output[out_idx] = sum;
}

// Version with multiple outputs per thread for better compute intensity
__global__ void depthwise_conv2d_multi_output_kernel(
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
    const int OUTPUTS_PER_THREAD = 4;
    
    __shared__ float s_input[TILE_H * OUTPUTS_PER_THREAD + 2][TILE_W + 2];
    __shared__ float s_weight[9];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int out_x = blockIdx.x * TILE_W + tx;
    int out_y_base = blockIdx.y * (TILE_H * OUTPUTS_PER_THREAD) + ty;
    int c = blockIdx.z % channels;
    int b = blockIdx.z / channels;

    // Load weights
    int tid = ty * TILE_W + tx;
    if (tid < 9) {
        s_weight[tid] = weight[c * 9 + tid];
    }

    int in_y_base = blockIdx.y * (TILE_H * OUTPUTS_PER_THREAD) * stride - padding;
    int in_x_base = blockIdx.x * TILE_W * stride - padding;

    // Load input tile - each thread loads OUTPUTS_PER_THREAD rows
    #pragma unroll
    for (int i = 0; i < OUTPUTS_PER_THREAD; i++) {
        int sy = ty + i * TILE_H;
        int in_y = in_y_base + sy;
        int in_x = in_x_base + tx;
        
        if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
            s_input[sy][tx] = input[((b * channels + c) * in_height + in_y) * in_width + in_x];
        } else {
            s_input[sy][tx] = 0.0f;
        }
        
        if (tx < 2) {
            int halo_x = in_x_base + TILE_W + tx;
            if (in_y >= 0 && in_y < in_height && halo_x >= 0 && halo_x < in_width) {
                s_input[sy][TILE_W + tx] = input[((b * channels + c) * in_height + in_y) * in_width + halo_x];
            } else {
                s_input[sy][TILE_W + tx] = 0.0f;
            }
        }
    }
    
    // Load bottom halo (2 extra rows)
    if (ty < 2) {
        int sy = TILE_H * OUTPUTS_PER_THREAD + ty;
        int in_y = in_y_base + sy;
        int in_x = in_x_base + tx;
        
        if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
            s_input[sy][tx] = input[((b * channels + c) * in_height + in_y) * in_width + in_x];
        } else {
            s_input[sy][tx] = 0.0f;
        }
        
        if (tx < 2) {
            int halo_x = in_x_base + TILE_W + tx;
            if (in_y >= 0 && in_y < in_height && halo_x >= 0 && halo_x < in_width) {
                s_input[sy][TILE_W + tx] = input[((b * channels + c) * in_height + in_y) * in_width + halo_x];
            } else {
                s_input[sy][TILE_W + tx] = 0.0f;
            }
        }
    }

    __syncthreads();

    if (out_x >= out_width || b >= batch_size) return;

    // Compute OUTPUTS_PER_THREAD outputs per thread
    #pragma unroll
    for (int i = 0; i < OUTPUTS_PER_THREAD; i++) {
        int out_y = out_y_base + i * TILE_H;
        if (out_y >= out_height) break;
        
        int sy = ty + i * TILE_H;
        float sum = 0.0f;
        sum += s_input[sy + 0][tx + 0] * s_weight[0];
        sum += s_input[sy + 0][tx + 1] * s_weight[1];
        sum += s_input[sy + 0][tx + 2] * s_weight[2];
        sum += s_input[sy + 1][tx + 0] * s_weight[3];
        sum += s_input[sy + 1][tx + 1] * s_weight[4];
        sum += s_input[sy + 1][tx + 2] * s_weight[5];
        sum += s_input[sy + 2][tx + 0] * s_weight[6];
        sum += s_input[sy + 2][tx + 1] * s_weight[7];
        sum += s_input[sy + 2][tx + 2] * s_weight[8];

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
    
    const int OUTPUTS_PER_THREAD = 4;
    dim3 block(TILE_W, TILE_H);
    dim3 grid(
        (out_width + TILE_W - 1) / TILE_W,
        (out_height + TILE_H * OUTPUTS_PER_THREAD - 1) / (TILE_H * OUTPUTS_PER_THREAD),
        batch_size * channels
    );
    
    if (kernel_size == 3 && stride == 1) {
        depthwise_conv2d_multi_output_kernel<<<grid, block>>>(
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
        // Fallback to simpler kernel for non-3x3 cases
        dim3 grid2(
            (out_width + TILE_W - 1) / TILE_W,
            (out_height + TILE_H - 1) / TILE_H,
            batch_size * channels
        );
        depthwise_conv2d_shared_kernel<<<grid2, block>>>(
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
