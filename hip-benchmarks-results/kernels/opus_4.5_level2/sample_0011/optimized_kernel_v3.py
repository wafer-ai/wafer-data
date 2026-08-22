import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# More optimized fused kernel with shared memory for scales
fused_scale_maxpool_clamp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <float.h>

#define BLOCK_X 32
#define BLOCK_Y 8
#define POOL_SIZE 4

__global__ void fused_scale_maxpool_clamp_kernel_v3(
    const float* __restrict__ input,
    const float* __restrict__ scale,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const float clamp_min,
    const float clamp_max
) {
    // Shared memory for scale values
    __shared__ float s_scale[64];  // Max 64 channels
    
    // Load scales to shared memory
    if (threadIdx.x < channels && threadIdx.y == 0) {
        s_scale[threadIdx.x] = scale[threadIdx.x];
    }
    __syncthreads();
    
    int ow = blockIdx.x * BLOCK_X + threadIdx.x;
    int oh = blockIdx.y * BLOCK_Y + threadIdx.y;
    int bc = blockIdx.z;
    
    if (ow >= out_width || oh >= out_height) return;
    
    int c = bc % channels;
    int b = bc / channels;
    
    // Get scale from shared memory
    float s = s_scale[c];
    
    // Compute max over pooling window
    float max_val = -FLT_MAX;
    
    int h_start = oh * POOL_SIZE;
    int w_start = ow * POOL_SIZE;
    
    // Base pointer for this batch and channel
    const float* in_ptr = input + (b * channels + c) * in_height * in_width;
    
    // Unrolled 4x4 pooling
    #pragma unroll
    for (int dh = 0; dh < POOL_SIZE; dh++) {
        int h_idx = (h_start + dh) * in_width + w_start;
        #pragma unroll
        for (int dw = 0; dw < POOL_SIZE; dw++) {
            float val = in_ptr[h_idx + dw] * s;
            max_val = fmaxf(max_val, val);
        }
    }
    
    // Clamp
    max_val = fminf(fmaxf(max_val, clamp_min), clamp_max);
    
    int out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
    output[out_idx] = max_val;
}

torch::Tensor fused_scale_maxpool_clamp_hip(
    torch::Tensor input,
    torch::Tensor scale,
    int pool_size,
    float clamp_min,
    float clamp_max
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    
    const int out_height = in_height / pool_size;
    const int out_width = in_width / pool_size;
    
    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());
    
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid(
        (out_width + BLOCK_X - 1) / BLOCK_X,
        (out_height + BLOCK_Y - 1) / BLOCK_Y,
        batch_size * channels
    );
    
    fused_scale_maxpool_clamp_kernel_v3<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        scale.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_height,
        in_width,
        out_height,
        out_width,
        clamp_min,
        clamp_max
    );
    
    return output;
}
"""

fused_scale_maxpool_clamp_cpp = """
torch::Tensor fused_scale_maxpool_clamp_hip(
    torch::Tensor input,
    torch::Tensor scale,
    int pool_size,
    float clamp_min,
    float clamp_max
);
"""

fused_module = load_inline(
    name="fused_scale_maxpool_clamp_v3",
    cpp_sources=fused_scale_maxpool_clamp_cpp,
    cuda_sources=fused_scale_maxpool_clamp_source,
    functions=["fused_scale_maxpool_clamp_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized model with fused Scale + MaxPool + Clamp kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.fused_module = fused_module

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            Output tensor of shape (batch_size, out_channels, height', width').
        """
        x = self.conv(x)
        x = self.group_norm(x)
        # Fused Scale + MaxPool + Clamp
        x = self.fused_module.fused_scale_maxpool_clamp_hip(
            x.contiguous(),
            self.scale.view(-1).contiguous(),
            self.maxpool_kernel_size,
            self.clamp_min,
            self.clamp_max
        )
        return x


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128 
kernel_size = 3
num_groups = 16
scale_shape = (out_channels, 1, 1)
maxpool_kernel_size = 4
clamp_min = 0.0
clamp_max = 1.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max]


def custom_kernel(inputs):
    model = ModelNew(*get_init_inputs()).cuda()
    return model(inputs[0])
