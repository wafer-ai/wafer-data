import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized kernel using 2D blocks and better occupancy
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 8

// 2D grid kernel with better occupancy
__global__ __launch_bounds__(256) void fused_subtract_tanh_subtract_avgpool_2d(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_channels,  // batch_size * channels
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const float sub1,
    const float sub2
) {
    int ow = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
    int oh = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;
    int bc = blockIdx.z;  // batch * channel combined
    
    if (ow >= out_width || oh >= out_height) return;
    
    int ih = oh * 2;
    int iw = ow * 2;
    
    const int in_hw = in_height * in_width;
    const int out_hw = out_height * out_width;
    
    const float* in_base = input + bc * in_hw + ih * in_width + iw;
    
    // Load 2x2 window
    float v00 = in_base[0];
    float v01 = in_base[1];
    float v10 = in_base[in_width];
    float v11 = in_base[in_width + 1];
    
    // Fused ops: tanh(x - sub1) - sub2
    v00 = tanhf(v00 - sub1) - sub2;
    v01 = tanhf(v01 - sub1) - sub2;
    v10 = tanhf(v10 - sub1) - sub2;
    v11 = tanhf(v11 - sub1) - sub2;
    
    // Average pool
    float result = (v00 + v01 + v10 + v11) * 0.25f;
    
    output[bc * out_hw + oh * out_width + ow] = result;
}

// Vector4 version processing 4 outputs per thread
__global__ __launch_bounds__(256) void fused_subtract_tanh_subtract_avgpool_vec4_2d(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const float sub1,
    const float sub2
) {
    // Each thread processes 4 consecutive output elements
    int ow_base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int bc = blockIdx.z;
    
    if (ow_base >= out_width || oh >= out_height) return;
    
    int ih = oh * 2;
    int iw_base = ow_base * 2;
    
    const int in_hw = in_height * in_width;
    const int out_hw = out_height * out_width;
    
    const float* row0 = input + bc * in_hw + ih * in_width + iw_base;
    const float* row1 = row0 + in_width;
    
    float4 out;
    
    // Process 4 output pixels
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int col = i * 2;
        float v00 = tanhf(row0[col] - sub1) - sub2;
        float v01 = tanhf(row0[col + 1] - sub1) - sub2;
        float v10 = tanhf(row1[col] - sub1) - sub2;
        float v11 = tanhf(row1[col + 1] - sub1) - sub2;
        
        float result = (v00 + v01 + v10 + v11) * 0.25f;
        
        if (i == 0) out.x = result;
        else if (i == 1) out.y = result;
        else if (i == 2) out.z = result;
        else out.w = result;
    }
    
    // Store as float4
    float4* out_ptr = (float4*)(output + bc * out_hw + oh * out_width + ow_base);
    *out_ptr = out;
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
    const int batch_channels = batch_size * channels;
    
    auto output = torch::empty({batch_size, channels, out_height, out_width}, 
                               input.options());
    
    if (pool_size == 2 && out_width % 4 == 0) {
        // Use vectorized 2D kernel
        dim3 block(BLOCK_DIM_X / 4, BLOCK_DIM_Y);  // Each thread does 4 outputs
        dim3 grid((out_width / 4 + block.x - 1) / block.x,
                  (out_height + block.y - 1) / block.y,
                  batch_channels);
        
        fused_subtract_tanh_subtract_avgpool_vec4_2d<<<grid, block>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_channels, in_height, in_width,
            out_height, out_width,
            subtract1_value, subtract2_value
        );
    } else {
        dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
        dim3 grid((out_width + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
                  (out_height + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y,
                  batch_channels);
        
        fused_subtract_tanh_subtract_avgpool_2d<<<grid, block>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_channels, in_height, in_width,
            out_height, out_width,
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
    extra_cuda_cflags=["-O3"]
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
