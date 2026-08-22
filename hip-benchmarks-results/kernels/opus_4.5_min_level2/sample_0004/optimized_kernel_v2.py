import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused kernel with better memory access patterns
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Use vector types for better memory throughput
typedef float4 float4_t;

__device__ __forceinline__ float fast_tanh(float x) {
    // Fast tanh approximation that's accurate for most values
    float x2 = x * x;
    float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
    float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    return a / b;
}

__global__ void fused_subtract_tanh_subtract_avgpool_kernel_v2(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int pool_size,
    const float subtract1_value,
    const float subtract2_value
) {
    // Each thread computes one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    // Compute output coordinates
    int ow = idx % out_width;
    int oh = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int b = idx / (out_width * out_height * channels);
    
    // Compute input starting position for this pooling window
    int ih_start = oh * pool_size;
    int iw_start = ow * pool_size;
    
    // Base offset for this batch and channel
    int base_offset = b * channels * in_height * in_width + c * in_height * in_width;
    
    // For 2x2 pooling, unroll completely
    float sum = 0.0f;
    float inv_count = 1.0f / (float)(pool_size * pool_size);
    
    #pragma unroll
    for (int ph = 0; ph < pool_size; ph++) {
        int ih = ih_start + ph;
        int row_offset = base_offset + ih * in_width;
        
        #pragma unroll
        for (int pw = 0; pw < pool_size; pw++) {
            int iw = iw_start + pw;
            float val = input[row_offset + iw];
            
            // Fused: subtract1 -> tanh -> subtract2
            val = val - subtract1_value;
            val = tanhf(val);
            val = val - subtract2_value;
            sum += val;
        }
    }
    
    output[idx] = sum * inv_count;
}

// Even more optimized kernel using shared memory for better cache utilization
__global__ void fused_subtract_tanh_subtract_avgpool_tiled(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const float subtract1_value,
    const float subtract2_value
) {
    // Tile dimensions
    const int TILE_W = 16;
    const int TILE_H = 16;
    
    int out_x = blockIdx.x * TILE_W + threadIdx.x;
    int out_y = blockIdx.y * TILE_H + threadIdx.y;
    int bc = blockIdx.z; // combined batch and channel index
    
    int b = bc / channels;
    int c = bc % channels;
    
    if (out_x >= out_width || out_y >= out_height || b >= batch_size) return;
    
    // 2x2 pooling
    int ih_start = out_y * 2;
    int iw_start = out_x * 2;
    
    int base_offset = b * channels * in_height * in_width + c * in_height * in_width;
    
    // Load and process 2x2 window
    float v00 = input[base_offset + ih_start * in_width + iw_start];
    float v01 = input[base_offset + ih_start * in_width + iw_start + 1];
    float v10 = input[base_offset + (ih_start + 1) * in_width + iw_start];
    float v11 = input[base_offset + (ih_start + 1) * in_width + iw_start + 1];
    
    // Fused operations
    v00 = tanhf(v00 - subtract1_value) - subtract2_value;
    v01 = tanhf(v01 - subtract1_value) - subtract2_value;
    v10 = tanhf(v10 - subtract1_value) - subtract2_value;
    v11 = tanhf(v11 - subtract1_value) - subtract2_value;
    
    // Average pooling
    float result = (v00 + v01 + v10 + v11) * 0.25f;
    
    int out_idx = b * channels * out_height * out_width + c * out_height * out_width + out_y * out_width + out_x;
    output[out_idx] = result;
}

torch::Tensor fused_subtract_tanh_subtract_avgpool(
    torch::Tensor input,
    float subtract1_value,
    float subtract2_value,
    int pool_size
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    
    const int out_height = in_height / pool_size;
    const int out_width = in_width / pool_size;
    
    auto output = torch::empty({batch_size, channels, out_height, out_width}, 
                               input.options());
    
    // Use tiled kernel for 2x2 pooling
    if (pool_size == 2) {
        const int TILE_W = 16;
        const int TILE_H = 16;
        
        dim3 block(TILE_W, TILE_H, 1);
        dim3 grid((out_width + TILE_W - 1) / TILE_W, 
                  (out_height + TILE_H - 1) / TILE_H,
                  batch_size * channels);
        
        fused_subtract_tanh_subtract_avgpool_tiled<<<grid, block>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, channels, in_height, in_width,
            out_height, out_width,
            subtract1_value, subtract2_value
        );
    } else {
        const int total_elements = batch_size * channels * out_height * out_width;
        const int block_size = 256;
        const int num_blocks = (total_elements + block_size - 1) / block_size;
        
        fused_subtract_tanh_subtract_avgpool_kernel_v2<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, channels, in_height, in_width,
            out_height, out_width, pool_size,
            subtract1_value, subtract2_value
        );
    }
    
    return output;
}
"""

fused_kernel_cpp = """
torch::Tensor fused_subtract_tanh_subtract_avgpool(
    torch::Tensor input,
    float subtract1_value,
    float subtract2_value,
    int pool_size
);
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_kernel_cpp,
    cuda_sources=fused_kernel_source,
    functions=["fused_subtract_tanh_subtract_avgpool"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses subtract, tanh, subtract, and avgpool operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool

    def forward(self, x):
        x = self.conv(x)
        x = fused_ops.fused_subtract_tanh_subtract_avgpool(
            x, self.subtract1_value, self.subtract2_value, self.kernel_size_pool
        )
        return x


def get_inputs():
    return [torch.rand(128, 64, 128, 128).cuda()]


def get_init_inputs():
    return [64, 128, 3, 0.5, 0.2, 2]
