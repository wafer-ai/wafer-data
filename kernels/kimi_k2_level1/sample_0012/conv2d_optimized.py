import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Define an optimized HIP kernel for 2D convolution
# This version uses shared memory for weights and has better memory coalescing
conv2d_hip_source = """
#include <hip/hip_runtime.h>

#define TILE_WIDTH 16
#define KERNEL_SIZE 3
#define UNROLL_FACTOR 4

__global__ void conv2d_optimized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height_in,
    int width_in,
    int height_out,
    int width_out
) {
    // Shared memory for kernel weights - small enough to fit in L1 cache
    __shared__ float weight_tile[KERNEL_SIZE][KERNEL_SIZE];
    
    // Batch and output channel for this thread block
    int b = blockIdx.z / out_channels;
    int oc = blockIdx.z % out_channels;
    
    // Output position this block computes
    int row_out_base = blockIdx.y * TILE_WIDTH;
    int col_out_base = blockIdx.x * TILE_WIDTH;
    
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    
    // Load kernel weights into shared memory (all threads load same weights)
    if (thread_row < KERNEL_SIZE && thread_col < KERNEL_SIZE) {
        weight_tile[thread_row][thread_col] = weight[(oc * in_channels + 0) * KERNEL_SIZE * KERNEL_SIZE + 
                                                      thread_row * KERNEL_SIZE + thread_col];
    }
    
    __syncthreads();
    
    // Each thread computes one output element
    int row_out = row_out_base + thread_row;
    int col_out = col_out_base + thread_col;
    
    if (row_out >= height_out || col_out >= width_out) return;
    
    float sum = 0.0f;
    
    // Process input channels in batches for better cache locality
    for (int ic_base = 0; ic_base < in_channels; ic_base += UNROLL_FACTOR) {
        
        // Unroll over UNROLL_FACTOR input channels
        #pragma unroll
        for (int ic_offset = 0; ic_offset < UNROLL_FACTOR; ic_offset++) {
            int ic = ic_base + ic_offset;
            if (ic >= in_channels) break;
            
            // Get pointer to input for this channel
            // Coalesced memory access: threads in a warp access consecutive memory
            const float* input_channel = &input[((b * in_channels + ic) * height_in + row_out) * width_in + col_out];
            const float* kernel_channel = &weight[(oc * in_channels + ic) * KERNEL_SIZE * KERNEL_SIZE];
            
            // Accumulate convolution
            #pragma unroll
            for (int kr = 0; kr < KERNEL_SIZE; kr++) {
                #pragma unroll
                for (int kc = 0; kc < KERNEL_SIZE; kc++) {
                    float input_val = input_channel[kr * width_in + kc];
                    float weight_val = kernel_channel[kr * KERNEL_SIZE + kc];
                    sum += input_val * weight_val;
                }
            }
        }
    }
    
    // Write output with coalesced access
    int output_idx = ((b * out_channels + oc) * height_out + row_out) * width_out + col_out;
    output[output_idx] = sum;
}

torch::Tensor conv2d_hip_forward(
    torch::Tensor input,
    torch::Tensor weight
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height_in = input.size(2);
    const int width_in = input.size(3);
    
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    // Calculate output dimensions (no padding, stride=1)
    const int height_out = height_in - kernel_size + 1;
    const int width_out = width_in - kernel_size + 1;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, height_out, width_out}, 
                               input.options());
    
    // Define grid and block dimensions
    dim3 block_dim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 grid_dim(
        (width_out + TILE_WIDTH - 1) / TILE_WIDTH,
        (height_out + TILE_WIDTH - 1) / TILE_WIDTH,
        batch_size * out_channels
    );
    
    // Launch kernel
    hipLaunchKernelGGL(
        conv2d_optimized_kernel,
        grid_dim,
        block_dim,
        0, 0,
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height_in,
        width_in,
        height_out,
        width_out
    );
    
    return output;
}
"""

# Compile the HIP kernel
conv2d_hip = load_inline(
    name='conv2d_hip',
    cpp_sources=conv2d_hip_source,
    functions=['conv2d_hip_forward'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        assert stride == 1, "Only stride=1 is supported"
        assert padding == 0, "Only padding=0 is supported"
        assert dilation == 1, "Only dilation=1 is supported"
        assert groups == 1, "Only groups=1 is supported"
        assert bias == False, "Only bias=False is supported"
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.kernel_size = kernel_size
        self.conv2d_hip = conv2d_hip
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, "Input must be 4D tensor (batch, channels, height, width)"
        assert x.size(1) == self.weight.size(1), "Input channels must match weight channels"
        assert self.kernel_size == 3, "Only kernel_size=3 is currently optimized"
        
        return self.conv2d_hip.conv2d_hip_forward(x, self.weight)

def get_inputs():
    batch_size = 16
    in_channels = 16
    height = 1024
    width = 1024
    x = torch.rand(batch_size, in_channels, height, width, device='cuda', dtype=torch.float32)
    return [x]

def get_init_inputs():
    in_channels = 16
    out_channels = 128
    kernel_size = 3
    return [in_channels, out_channels, kernel_size]
