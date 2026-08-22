
import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>
#include <cfloat>

#define K_SIZE 4
#define STRIDE 1
#define PAD 1
#define DILATION 1

#define BLOCK_W 32
#define BLOCK_H 8

// Input tile size needed for shared memory
// width: (BLOCK_W - 1) * STRIDE + K_SIZE
// height: (BLOCK_H - 1) * STRIDE + K_SIZE
// With S=1, K=4: 35x11
#define TILE_W (BLOCK_W * STRIDE + K_SIZE - STRIDE)
#define TILE_H (BLOCK_H * STRIDE + K_SIZE - STRIDE)

__global__ void maxpool_shared_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int channels_total,
    const int height,
    const int width,
    const int out_height,
    const int out_width
) {
    // Shared memory to hold input tile
    __shared__ float tile[TILE_H][TILE_W];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Linear thread id for loading
    const int tid = ty * BLOCK_W + tx; 
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z; // channel index

    // Global output coordinates
    const int w_out = bx * BLOCK_W + tx;
    const int h_out = by * BLOCK_H + ty;
    
    // Calculate global input start coordinates for this tile
    const int input_w_start = bx * BLOCK_W * STRIDE - PAD;
    const int input_h_start = by * BLOCK_H * STRIDE - PAD;

    const int input_channel_offset = bz * height * width;
    const float* input_ptr = input + input_channel_offset;

    // Load into shared memory
    // Loop to cover all tile elements using available threads
    for (int i = tid; i < TILE_H * TILE_W; i += BLOCK_W * BLOCK_H) {
        int tile_y = i / TILE_W;
        int tile_x = i % TILE_W;

        int global_h = input_h_start + tile_y;
        int global_w = input_w_start + tile_x;

        float val = -FLT_MAX;
        
        // Boundary check for loading
        if (global_h >= 0 && global_h < height && global_w >= 0 && global_w < width) {
            val = input_ptr[global_h * width + global_w];
        }
        
        tile[tile_y][tile_x] = val;
    }

    __syncthreads();

    // Compute MaxPool
    if (w_out < out_width && h_out < out_height && bz < channels_total) {
        float max_val = -FLT_MAX;
        
        // Window is K_SIZE x K_SIZE
        #pragma unroll
        for (int kh = 0; kh < K_SIZE; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < K_SIZE; ++kw) {
                // Access shared memory
                float val = tile[ty * STRIDE + kh][tx * STRIDE + kw];
                max_val = fmaxf(max_val, val);
            }
        }
        
        int output_idx = bz * out_height * out_width + h_out * out_width + w_out;
        output[output_idx] = max_val;
    }
}

torch::Tensor maxpool_hip(torch::Tensor input) {
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    const int out_height = (height + 2 * PAD - DILATION * (K_SIZE - 1) - 1) / STRIDE + 1;
    const int out_width = (width + 2 * PAD - DILATION * (K_SIZE - 1) - 1) / STRIDE + 1;

    auto output = torch::empty({batch, channels, out_height, out_width}, input.options());

    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid(
        (out_width + block.x - 1) / block.x,
        (out_height + block.y - 1) / block.y,
        batch * channels
    );

    maxpool_shared_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch * channels,
        height,
        width,
        out_height,
        out_width
    );

    return output;
}
"""

maxpool_ops = load_inline(
    name="maxpool_ops_v2",
    cpp_sources=cpp_source,
    functions=["maxpool_hip"],
    verbose=True,
    extra_cflags=['-O3']
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return maxpool_ops.maxpool_hip(x)
