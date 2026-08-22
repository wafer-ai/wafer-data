import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

depthwise_conv_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define TILE_W 16
#define TILE_H 16

// Optimized depthwise convolution kernel with shared memory tiling
// Specialized for 3x3 kernel
template<int KERNEL_SIZE>
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
    int stride,
    int padding
) {
    // Each block handles one tile of one channel of one batch
    int bc = blockIdx.z;  // combined batch and channel index
    int batch = bc / channels;
    int channel = bc % channels;
    
    int tile_x = blockIdx.x * TILE_W;
    int tile_y = blockIdx.y * TILE_H;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int out_x = tile_x + tx;
    int out_y = tile_y + ty;
    
    // Shared memory for input tile (including halo for kernel)
    __shared__ float smem_input[(TILE_H + KERNEL_SIZE - 1)][(TILE_W + KERNEL_SIZE - 1)];
    __shared__ float smem_weight[KERNEL_SIZE][KERNEL_SIZE];
    
    // Load weights into shared memory (only need threads in the kernel size)
    if (tx < KERNEL_SIZE && ty < KERNEL_SIZE) {
        smem_weight[ty][tx] = weight[channel * KERNEL_SIZE * KERNEL_SIZE + ty * KERNEL_SIZE + tx];
    }
    
    // Calculate the input region we need
    int in_tile_x = tile_x * stride - padding;
    int in_tile_y = tile_y * stride - padding;
    
    // Load input tile into shared memory with halo
    // Each thread loads one or more elements
    int smem_h = TILE_H + KERNEL_SIZE - 1;
    int smem_w = TILE_W + KERNEL_SIZE - 1;
    
    for (int dy = ty; dy < smem_h; dy += TILE_H) {
        for (int dx = tx; dx < smem_w; dx += TILE_W) {
            int in_y = in_tile_y + dy;
            int in_x = in_tile_x + dx;
            
            float val = 0.0f;
            if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                int in_idx = ((batch * channels + channel) * in_height + in_y) * in_width + in_x;
                val = input[in_idx];
            }
            smem_input[dy][dx] = val;
        }
    }
    
    __syncthreads();
    
    // Compute convolution for this output pixel
    if (out_x < out_width && out_y < out_height) {
        float sum = 0.0f;
        
        #pragma unroll
        for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
            #pragma unroll
            for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                sum += smem_input[ty * stride + ky][tx * stride + kx] * smem_weight[ky][kx];
            }
        }
        
        int out_idx = ((batch * channels + channel) * out_height + out_y) * out_width + out_x;
        output[out_idx] = sum;
    }
}

// Generic kernel for arbitrary kernel sizes
__global__ void depthwise_conv2d_kernel_generic(
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
    
    int out_idx = ((batch * channels + channel) * out_height + out_y) * out_width + out_x;
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
    
    if (kernel_size == 3) {
        depthwise_conv2d_kernel<3><<<grid, block>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, channels,
            in_height, in_width,
            out_height, out_width,
            stride, padding
        );
    } else if (kernel_size == 5) {
        depthwise_conv2d_kernel<5><<<grid, block>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, channels,
            in_height, in_width,
            out_height, out_width,
            stride, padding
        );
    } else {
        depthwise_conv2d_kernel_generic<<<grid, block>>>(
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
