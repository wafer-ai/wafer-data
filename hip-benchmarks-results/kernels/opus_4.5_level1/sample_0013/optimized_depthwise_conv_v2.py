import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

depthwise_conv_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define TILE_W 32
#define TILE_H 8

// Optimized depthwise convolution kernel with shared memory tiling
// Each thread computes multiple output elements for better arithmetic intensity
template<int KERNEL_SIZE>
__global__ void depthwise_conv2d_kernel_v2(
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
    constexpr int smem_h = TILE_H + KERNEL_SIZE - 1;
    constexpr int smem_w = TILE_W + KERNEL_SIZE - 1;
    __shared__ float smem_input[smem_h][smem_w];
    
    // Load weights into registers
    float w[KERNEL_SIZE][KERNEL_SIZE];
    const float* weight_ptr = weight + channel * KERNEL_SIZE * KERNEL_SIZE;
    
    #pragma unroll
    for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
        #pragma unroll
        for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
            w[ky][kx] = weight_ptr[ky * KERNEL_SIZE + kx];
        }
    }
    
    // Calculate the input region we need
    int in_tile_x = tile_x * stride - padding;
    int in_tile_y = tile_y * stride - padding;
    
    const float* input_base = input + (batch * channels + channel) * in_height * in_width;
    
    // Load input tile into shared memory with halo
    // Use all threads to load cooperatively
    int thread_id = ty * TILE_W + tx;
    int total_threads = TILE_W * TILE_H;
    int total_elements = smem_h * smem_w;
    
    for (int idx = thread_id; idx < total_elements; idx += total_threads) {
        int sy = idx / smem_w;
        int sx = idx % smem_w;
        
        int in_y = in_tile_y + sy;
        int in_x = in_tile_x + sx;
        
        float val = 0.0f;
        if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
            val = input_base[in_y * in_width + in_x];
        }
        smem_input[sy][sx] = val;
    }
    
    __syncthreads();
    
    // Compute convolution for this output pixel
    if (out_x < out_width && out_y < out_height) {
        float sum = 0.0f;
        
        int local_y = ty * stride;
        int local_x = tx * stride;
        
        #pragma unroll
        for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
            #pragma unroll
            for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                sum += smem_input[local_y + ky][local_x + kx] * w[ky][kx];
            }
        }
        
        int out_idx = ((batch * channels + channel) * out_height + out_y) * out_width + out_x;
        output[out_idx] = sum;
    }
}

// Highly optimized kernel for 3x3 specifically
__global__ void depthwise_conv2d_3x3_optimized(
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
    constexpr int KERNEL_SIZE = 3;
    constexpr int TILE_W_OPT = 32;
    constexpr int TILE_H_OPT = 8;
    constexpr int smem_h = TILE_H_OPT + 2;
    constexpr int smem_w = TILE_W_OPT + 2;
    
    int bc = blockIdx.z;
    int batch = bc / channels;
    int channel = bc % channels;
    
    int tile_x = blockIdx.x * TILE_W_OPT;
    int tile_y = blockIdx.y * TILE_H_OPT;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int out_x = tile_x + tx;
    int out_y = tile_y + ty;
    
    __shared__ float smem_input[smem_h][smem_w + 1];  // +1 to avoid bank conflicts
    
    // Load weights into registers
    float w0, w1, w2, w3, w4, w5, w6, w7, w8;
    const float* weight_ptr = weight + channel * 9;
    w0 = weight_ptr[0]; w1 = weight_ptr[1]; w2 = weight_ptr[2];
    w3 = weight_ptr[3]; w4 = weight_ptr[4]; w5 = weight_ptr[5];
    w6 = weight_ptr[6]; w7 = weight_ptr[7]; w8 = weight_ptr[8];
    
    int in_tile_x = tile_x * stride - padding;
    int in_tile_y = tile_y * stride - padding;
    
    const float* input_base = input + (batch * channels + channel) * in_height * in_width;
    
    // Cooperative loading
    int thread_id = ty * TILE_W_OPT + tx;
    int total_threads = TILE_W_OPT * TILE_H_OPT;
    int total_elements = smem_h * smem_w;
    
    for (int idx = thread_id; idx < total_elements; idx += total_threads) {
        int sy = idx / smem_w;
        int sx = idx % smem_w;
        
        int in_y = in_tile_y + sy;
        int in_x = in_tile_x + sx;
        
        float val = 0.0f;
        if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
            val = input_base[in_y * in_width + in_x];
        }
        smem_input[sy][sx] = val;
    }
    
    __syncthreads();
    
    if (out_x < out_width && out_y < out_height) {
        int ly = ty * stride;
        int lx = tx * stride;
        
        float sum = smem_input[ly][lx] * w0
                  + smem_input[ly][lx+1] * w1
                  + smem_input[ly][lx+2] * w2
                  + smem_input[ly+1][lx] * w3
                  + smem_input[ly+1][lx+1] * w4
                  + smem_input[ly+1][lx+2] * w5
                  + smem_input[ly+2][lx] * w6
                  + smem_input[ly+2][lx+1] * w7
                  + smem_input[ly+2][lx+2] * w8;
        
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
    
    if (kernel_size == 3) {
        dim3 block(32, 8);
        dim3 grid(
            (out_width + 31) / 32,
            (out_height + 7) / 8,
            batch_size * channels
        );
        depthwise_conv2d_3x3_optimized<<<grid, block>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, channels,
            in_height, in_width,
            out_height, out_width,
            stride, padding
        );
    } else if (kernel_size == 5) {
        dim3 block(TILE_W, TILE_H);
        dim3 grid(
            (out_width + TILE_W - 1) / TILE_W,
            (out_height + TILE_H - 1) / TILE_H,
            batch_size * channels
        );
        depthwise_conv2d_kernel_v2<5><<<grid, block>>>(
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
