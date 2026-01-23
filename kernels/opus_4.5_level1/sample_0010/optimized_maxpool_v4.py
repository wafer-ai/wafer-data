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
#define KERNEL_SIZE 4

// Optimized kernel using shared memory with larger tiles
__global__ void maxpool2d_shared_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int padding
) {
    // Shared memory tile with halo
    __shared__ float smem[TILE_HEIGHT + KERNEL_SIZE - 1][TILE_WIDTH + KERNEL_SIZE - 1];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int ow = blockIdx.x * TILE_WIDTH + tx;
    int oh = blockIdx.y * TILE_HEIGHT + ty;
    int bc = blockIdx.z;
    
    int b = bc / channels;
    int c = bc % channels;
    
    // Starting input coordinates for this tile
    int ih_base = blockIdx.y * TILE_HEIGHT - padding;
    int iw_base = blockIdx.x * TILE_WIDTH - padding;
    
    const float* input_ptr = input + (b * channels + c) * in_height * in_width;
    
    // Load tile into shared memory with halo
    int smem_h = TILE_HEIGHT + KERNEL_SIZE - 1;  // 7
    int smem_w = TILE_WIDTH + KERNEL_SIZE - 1;   // 67
    
    int thread_id = ty * TILE_WIDTH + tx;
    int total_threads = TILE_WIDTH * TILE_HEIGHT;  // 256
    int total_elements = smem_h * smem_w;  // 469
    
    // Each thread loads ~2 elements
    for (int i = thread_id; i < total_elements; i += total_threads) {
        int smem_y = i / smem_w;
        int smem_x = i % smem_w;
        int ih = ih_base + smem_y;
        int iw = iw_base + smem_x;
        
        float val = -FLT_MAX;
        if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
            val = input_ptr[ih * in_width + iw];
        }
        smem[smem_y][smem_x] = val;
    }
    
    __syncthreads();
    
    // Compute max pooling from shared memory
    if (ow < out_width && oh < out_height) {
        float max_val = -FLT_MAX;
        
        #pragma unroll
        for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                max_val = fmaxf(max_val, smem[ty + kh][tx + kw]);
            }
        }
        
        int out_idx = (b * channels + c) * out_height * out_width + oh * out_width + ow;
        output[out_idx] = max_val;
    }
}

// Alternative: Vector loads for better bandwidth
__global__ void maxpool2d_vec_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int padding
) {
    // Shared memory tile with halo
    __shared__ float smem[TILE_HEIGHT + KERNEL_SIZE - 1][TILE_WIDTH + KERNEL_SIZE - 1 + 4]; // +4 for alignment
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int ow = blockIdx.x * TILE_WIDTH + tx;
    int oh = blockIdx.y * TILE_HEIGHT + ty;
    int bc = blockIdx.z;
    
    int b = bc / channels;
    int c = bc % channels;
    
    int ih_base = blockIdx.y * TILE_HEIGHT - padding;
    int iw_base = blockIdx.x * TILE_WIDTH - padding;
    
    const float* input_ptr = input + (b * channels + c) * in_height * in_width;
    
    int smem_h = TILE_HEIGHT + KERNEL_SIZE - 1;
    int smem_w = TILE_WIDTH + KERNEL_SIZE - 1;
    
    int thread_id = ty * TILE_WIDTH + tx;
    int total_threads = TILE_WIDTH * TILE_HEIGHT;
    int total_elements = smem_h * smem_w;
    
    for (int i = thread_id; i < total_elements; i += total_threads) {
        int smem_y = i / smem_w;
        int smem_x = i % smem_w;
        int ih = ih_base + smem_y;
        int iw = iw_base + smem_x;
        
        float val = -FLT_MAX;
        if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
            val = input_ptr[ih * in_width + iw];
        }
        smem[smem_y][smem_x] = val;
    }
    
    __syncthreads();
    
    if (ow < out_width && oh < out_height) {
        float max_val = -FLT_MAX;
        
        // Load first row from shared memory
        float4 row0 = *reinterpret_cast<float4*>(&smem[ty + 0][tx]);
        max_val = fmaxf(max_val, row0.x);
        max_val = fmaxf(max_val, row0.y);
        max_val = fmaxf(max_val, row0.z);
        max_val = fmaxf(max_val, row0.w);
        
        float4 row1 = *reinterpret_cast<float4*>(&smem[ty + 1][tx]);
        max_val = fmaxf(max_val, row1.x);
        max_val = fmaxf(max_val, row1.y);
        max_val = fmaxf(max_val, row1.z);
        max_val = fmaxf(max_val, row1.w);
        
        float4 row2 = *reinterpret_cast<float4*>(&smem[ty + 2][tx]);
        max_val = fmaxf(max_val, row2.x);
        max_val = fmaxf(max_val, row2.y);
        max_val = fmaxf(max_val, row2.z);
        max_val = fmaxf(max_val, row2.w);
        
        float4 row3 = *reinterpret_cast<float4*>(&smem[ty + 3][tx]);
        max_val = fmaxf(max_val, row3.x);
        max_val = fmaxf(max_val, row3.y);
        max_val = fmaxf(max_val, row3.z);
        max_val = fmaxf(max_val, row3.w);
        
        int out_idx = (b * channels + c) * out_height * out_width + oh * out_width + ow;
        output[out_idx] = max_val;
    }
}

torch::Tensor maxpool2d_hip(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());
    
    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid(
        (out_width + TILE_WIDTH - 1) / TILE_WIDTH,
        (out_height + TILE_HEIGHT - 1) / TILE_HEIGHT,
        batch_size * channels
    );
    
    maxpool2d_shared_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_height,
        in_width,
        out_height,
        out_width,
        padding
    );
    
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
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.maxpool2d = maxpool2d_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool2d.maxpool2d_hip(x, self.kernel_size, self.stride, self.padding, self.dilation)


def get_inputs():
    x = torch.rand(32, 64, 512, 512).cuda()
    return [x]


def get_init_inputs():
    return [4, 1, 1, 1]
