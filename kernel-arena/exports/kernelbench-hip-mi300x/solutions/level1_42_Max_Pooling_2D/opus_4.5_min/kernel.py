import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

maxpool2d_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <float.h>

#define TILE_WIDTH 64
#define TILE_HEIGHT 4

// Specialized kernel for kernel_size=4, stride=1, padding=1, dilation=1
// Uses shared memory with explicit memory access optimization
__global__ __launch_bounds__(256)
void maxpool2d_kernel_k4s1p1d1_opt(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width
) {
    // Shared memory for input tile with halo
    __shared__ float tile[TILE_HEIGHT + 3][TILE_WIDTH + 4];  // +4 for alignment
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    const int out_x_base = blockIdx.x * TILE_WIDTH;
    const int out_y_base = blockIdx.y * TILE_HEIGHT;
    const int bc = blockIdx.z;
    
    const int b = bc / channels;
    const int c = bc % channels;
    
    const int input_base = (b * channels + c) * input_height * input_width;
    
    // Input position = output position - padding = output position - 1
    const int in_x_base = out_x_base - 1;
    const int in_y_base = out_y_base - 1;
    
    // Tile dimensions to load
    const int tile_h = TILE_HEIGHT + 3;
    const int tile_w = TILE_WIDTH + 3;
    const int num_threads = TILE_WIDTH * TILE_HEIGHT;
    
    const int tid = ty * TILE_WIDTH + tx;
    
    // Load input tile into shared memory
    const int total_elems = tile_h * tile_w;
    for (int idx = tid; idx < total_elems; idx += num_threads) {
        int tile_y = idx / tile_w;
        int tile_x = idx % tile_w;
        int in_y = in_y_base + tile_y;
        int in_x = in_x_base + tile_x;
        
        float val = -FLT_MAX;
        if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
            val = input[input_base + in_y * input_width + in_x];
        }
        tile[tile_y][tile_x] = val;
    }
    
    __syncthreads();
    
    // Compute output
    const int out_x = out_x_base + tx;
    const int out_y = out_y_base + ty;
    
    if (out_x < output_width && out_y < output_height) {
        float max_val = -FLT_MAX;
        
        // Unrolled 4x4 max pooling - explicitly unroll for better ILP
        float r0 = tile[ty][tx];
        float r1 = tile[ty][tx+1];
        float r2 = tile[ty][tx+2];
        float r3 = tile[ty][tx+3];
        max_val = fmaxf(fmaxf(r0, r1), fmaxf(r2, r3));
        
        r0 = tile[ty+1][tx];
        r1 = tile[ty+1][tx+1];
        r2 = tile[ty+1][tx+2];
        r3 = tile[ty+1][tx+3];
        max_val = fmaxf(max_val, fmaxf(fmaxf(r0, r1), fmaxf(r2, r3)));
        
        r0 = tile[ty+2][tx];
        r1 = tile[ty+2][tx+1];
        r2 = tile[ty+2][tx+2];
        r3 = tile[ty+2][tx+3];
        max_val = fmaxf(max_val, fmaxf(fmaxf(r0, r1), fmaxf(r2, r3)));
        
        r0 = tile[ty+3][tx];
        r1 = tile[ty+3][tx+1];
        r2 = tile[ty+3][tx+2];
        r3 = tile[ty+3][tx+3];
        max_val = fmaxf(max_val, fmaxf(fmaxf(r0, r1), fmaxf(r2, r3)));
        
        const int output_idx = ((b * channels + c) * output_height + out_y) * output_width + out_x;
        output[output_idx] = max_val;
    }
}

// Alternative with different tile size - 32x8
#define TILE_WIDTH2 32
#define TILE_HEIGHT2 8

__global__ __launch_bounds__(256)
void maxpool2d_kernel_k4s1p1d1_v2(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width
) {
    __shared__ float tile[TILE_HEIGHT2 + 3][TILE_WIDTH2 + 4];  // +4 for alignment
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    const int out_x_base = blockIdx.x * TILE_WIDTH2;
    const int out_y_base = blockIdx.y * TILE_HEIGHT2;
    const int bc = blockIdx.z;
    
    const int b = bc / channels;
    const int c = bc % channels;
    
    const int input_base = (b * channels + c) * input_height * input_width;
    
    const int in_x_base = out_x_base - 1;
    const int in_y_base = out_y_base - 1;
    
    const int tile_h = TILE_HEIGHT2 + 3;
    const int tile_w = TILE_WIDTH2 + 3;
    const int num_threads = TILE_WIDTH2 * TILE_HEIGHT2;
    
    const int tid = ty * TILE_WIDTH2 + tx;
    
    const int total_elems = tile_h * tile_w;
    for (int idx = tid; idx < total_elems; idx += num_threads) {
        int tile_y = idx / tile_w;
        int tile_x = idx % tile_w;
        int in_y = in_y_base + tile_y;
        int in_x = in_x_base + tile_x;
        
        float val = -FLT_MAX;
        if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
            val = input[input_base + in_y * input_width + in_x];
        }
        tile[tile_y][tile_x] = val;
    }
    
    __syncthreads();
    
    const int out_x = out_x_base + tx;
    const int out_y = out_y_base + ty;
    
    if (out_x < output_width && out_y < output_height) {
        float max_val = -FLT_MAX;
        
        // Unrolled 4x4 max pooling
        float r0 = tile[ty][tx];
        float r1 = tile[ty][tx+1];
        float r2 = tile[ty][tx+2];
        float r3 = tile[ty][tx+3];
        max_val = fmaxf(fmaxf(r0, r1), fmaxf(r2, r3));
        
        r0 = tile[ty+1][tx];
        r1 = tile[ty+1][tx+1];
        r2 = tile[ty+1][tx+2];
        r3 = tile[ty+1][tx+3];
        max_val = fmaxf(max_val, fmaxf(fmaxf(r0, r1), fmaxf(r2, r3)));
        
        r0 = tile[ty+2][tx];
        r1 = tile[ty+2][tx+1];
        r2 = tile[ty+2][tx+2];
        r3 = tile[ty+2][tx+3];
        max_val = fmaxf(max_val, fmaxf(fmaxf(r0, r1), fmaxf(r2, r3)));
        
        r0 = tile[ty+3][tx];
        r1 = tile[ty+3][tx+1];
        r2 = tile[ty+3][tx+2];
        r3 = tile[ty+3][tx+3];
        max_val = fmaxf(max_val, fmaxf(fmaxf(r0, r1), fmaxf(r2, r3)));
        
        const int output_idx = ((b * channels + c) * output_height + out_y) * output_width + out_x;
        output[output_idx] = max_val;
    }
}

// Generic kernel for other configurations
__global__ void maxpool2d_kernel_generic(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int bc = blockIdx.z;
    
    if (out_x >= output_width || out_y >= output_height)
        return;
    
    const int b = bc / channels;
    const int c = bc % channels;
    
    const int in_y_start = out_y * stride - padding;
    const int in_x_start = out_x * stride - padding;
    
    const int input_base = (b * channels + c) * input_height * input_width;
    
    float max_val = -FLT_MAX;
    
    for (int ky = 0; ky < kernel_size; ++ky) {
        const int in_y = in_y_start + ky * dilation;
        if (in_y >= 0 && in_y < input_height) {
            const int row_offset = input_base + in_y * input_width;
            for (int kx = 0; kx < kernel_size; ++kx) {
                const int in_x = in_x_start + kx * dilation;
                if (in_x >= 0 && in_x < input_width) {
                    max_val = fmaxf(max_val, input[row_offset + in_x]);
                }
            }
        }
    }
    
    const int output_idx = ((b * channels + c) * output_height + out_y) * output_width + out_x;
    output[output_idx] = max_val;
}

torch::Tensor maxpool2d_hip(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    
    const int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());
    
    if (kernel_size == 4 && stride == 1 && padding == 1 && dilation == 1) {
        // Try the 32x8 tile version - better occupancy
        dim3 block(TILE_WIDTH2, TILE_HEIGHT2);
        dim3 grid(
            (output_width + TILE_WIDTH2 - 1) / TILE_WIDTH2,
            (output_height + TILE_HEIGHT2 - 1) / TILE_HEIGHT2,
            batch_size * channels
        );
        
        maxpool2d_kernel_k4s1p1d1_v2<<<grid, block>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width
        );
    } else {
        dim3 block(32, 8);
        dim3 grid(
            (output_width + block.x - 1) / block.x,
            (output_height + block.y - 1) / block.y,
            batch_size * channels
        );
        
        maxpool2d_kernel_generic<<<grid, block>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_size,
            stride,
            padding,
            dilation
        );
    }
    
    return output;
}
"""

maxpool2d_cpp_source = """
torch::Tensor maxpool2d_hip(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
);
"""

maxpool2d_module = load_inline(
    name="maxpool2d_hip",
    cpp_sources=maxpool2d_cpp_source,
    cuda_sources=maxpool2d_hip_source,
    functions=["maxpool2d_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return maxpool2d_module.maxpool2d_hip(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation
        )


batch_size = 32
channels = 64
height = 512
width = 512
kernel_size = 4
stride = 1
padding = 1
dilation = 1

def get_inputs():
    x = torch.rand(batch_size, channels, height, width).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]
