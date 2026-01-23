import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

depthwise_conv2d_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>

#define TILE_W 16
#define TILE_H 16

// Optimized version using shared memory for 3x3 kernel
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
    int kernel_size,
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

    // Load weights into shared memory
    int tid = ty * TILE_W + tx;
    if (tid < 9) {
        s_weight[tid] = weight[c * 9 + tid];
    }

    // Calculate input coordinates for this thread
    int in_y_base = blockIdx.y * TILE_H * stride - padding;
    int in_x_base = blockIdx.x * TILE_W * stride - padding;

    // Load input tile into shared memory (with halo)
    int in_y = in_y_base + ty;
    int in_x = in_x_base + tx;
    
    // Main region
    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
        s_input[ty][tx] = input[((b * channels + c) * in_height + in_y) * in_width + in_x];
    } else {
        s_input[ty][tx] = 0.0f;
    }

    // Load right halo (last 2 columns need extra data)
    if (tx < 2) {
        int halo_x = in_x_base + TILE_W + tx;
        if (in_y >= 0 && in_y < in_height && halo_x >= 0 && halo_x < in_width) {
            s_input[ty][TILE_W + tx] = input[((b * channels + c) * in_height + in_y) * in_width + halo_x];
        } else {
            s_input[ty][TILE_W + tx] = 0.0f;
        }
    }

    // Load bottom halo (last 2 rows need extra data)
    if (ty < 2) {
        int halo_y = in_y_base + TILE_H + ty;
        if (halo_y >= 0 && halo_y < in_height && in_x >= 0 && in_x < in_width) {
            s_input[TILE_H + ty][tx] = input[((b * channels + c) * in_height + halo_y) * in_width + in_x];
        } else {
            s_input[TILE_H + ty][tx] = 0.0f;
        }
    }

    // Load corner (bottom-right)
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

    // Compute convolution using shared memory
    float sum = 0.0f;
    
    #pragma unroll
    for (int ky = 0; ky < 3; ky++) {
        #pragma unroll
        for (int kx = 0; kx < 3; kx++) {
            sum += s_input[ty + ky][tx + kx] * s_weight[ky * 3 + kx];
        }
    }

    int out_idx = ((b * channels + c) * out_height + out_y) * out_width + out_x;
    output[out_idx] = sum;
}

// Generic kernel for other kernel sizes
__global__ void depthwise_conv2d_kernel(
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
    int c = blockIdx.z % channels;
    int b = blockIdx.z / channels;

    if (out_x >= out_width || out_y >= out_height || b >= batch_size) return;

    float sum = 0.0f;
    
    int in_y_start = out_y * stride - padding;
    int in_x_start = out_x * stride - padding;

    for (int ky = 0; ky < kernel_size; ky++) {
        int in_y = in_y_start + ky;
        if (in_y >= 0 && in_y < in_height) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int in_x = in_x_start + kx;
                if (in_x >= 0 && in_x < in_width) {
                    int in_idx = ((b * channels + c) * in_height + in_y) * in_width + in_x;
                    int w_idx = c * kernel_size * kernel_size + ky * kernel_size + kx;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }

    int out_idx = ((b * channels + c) * out_height + out_y) * out_width + out_x;
    output[out_idx] = sum;
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
    
    dim3 block(TILE_W, TILE_H);
    dim3 grid(
        (out_width + TILE_W - 1) / TILE_W,
        (out_height + TILE_H - 1) / TILE_H,
        batch_size * channels
    );
    
    if (kernel_size == 3 && stride == 1) {
        depthwise_conv2d_shared_kernel<<<grid, block>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            channels,
            in_height,
            in_width,
            out_height,
            out_width,
            kernel_size,
            stride,
            padding
        );
    } else {
        depthwise_conv2d_kernel<<<grid, block>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            channels,
            in_height,
            in_width,
            out_height,
            out_width,
            kernel_size,
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
        # Use standard nn.Conv2d for weight initialization (to get proper shape/init)
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get weight in (channels, 1, kH, kW) format, reshape to (channels, kH, kW)
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
