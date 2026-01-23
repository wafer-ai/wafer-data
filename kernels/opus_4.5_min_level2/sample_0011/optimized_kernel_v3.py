import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel for Scale + MaxPool + Clamp with float4 vectorization
fused_scale_maxpool_clamp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <float.h>

// Vectorized kernel for pool_size=4 using float4 loads
__global__ void fused_scale_maxpool_clamp_kernel_v3(
    const float* __restrict__ input,
    const float* __restrict__ scale,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    float clamp_min,
    float clamp_max
) {
    // Shared memory for scale values
    __shared__ float shared_scale[64]; // Assuming max 64 channels per block
    
    int block_start_ch = (blockIdx.x * blockDim.x) / (out_height * out_width);
    int block_end_ch = ((blockIdx.x + 1) * blockDim.x - 1) / (out_height * out_width);
    
    // Each thread computes one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_height * out_width;
    
    // Load scale values into shared memory (first few threads)
    if (threadIdx.x < channels && threadIdx.x < 64) {
        shared_scale[threadIdx.x] = scale[threadIdx.x];
    }
    __syncthreads();
    
    if (idx < total) {
        // Decode index
        int ow = idx % out_width;
        int oh = (idx / out_width) % out_height;
        int c = (idx / (out_width * out_height)) % channels;
        int b = idx / (out_width * out_height * channels);
        
        // Get scale value for this channel
        float scale_val = (c < 64) ? shared_scale[c] : scale[c];
        
        // Compute max pooling with scale and clamp
        float max_val = -FLT_MAX;
        
        int h_start = oh * 4;
        int w_start = ow * 4;
        
        // Base offset for this batch and channel
        int base_offset = b * (channels * in_height * in_width) + c * (in_height * in_width);
        
        // Unrolled loop for pool_size=4 with vectorized float4 loads
        #pragma unroll
        for (int ph = 0; ph < 4; ph++) {
            int ih = h_start + ph;
            int row_offset = base_offset + ih * in_width + w_start;
            
            // Use float4 to load 4 consecutive floats at once
            float4 vals = *reinterpret_cast<const float4*>(&input[row_offset]);
            
            // Find max of the 4 values, scaled
            float v0 = vals.x * scale_val;
            float v1 = vals.y * scale_val;
            float v2 = vals.z * scale_val;
            float v3 = vals.w * scale_val;
            
            max_val = fmaxf(max_val, v0);
            max_val = fmaxf(max_val, v1);
            max_val = fmaxf(max_val, v2);
            max_val = fmaxf(max_val, v3);
        }
        
        // Clamp the result
        max_val = fminf(fmaxf(max_val, clamp_min), clamp_max);
        
        output[idx] = max_val;
    }
}

// Generic kernel for arbitrary pool sizes
__global__ void fused_scale_maxpool_clamp_kernel_generic(
    const float* __restrict__ input,
    const float* __restrict__ scale,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int pool_size,
    float clamp_min,
    float clamp_max
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_height * out_width;
    
    if (idx < total) {
        int ow = idx % out_width;
        int oh = (idx / out_width) % out_height;
        int c = (idx / (out_width * out_height)) % channels;
        int b = idx / (out_width * out_height * channels);
        
        float scale_val = scale[c];
        float max_val = -FLT_MAX;
        
        int h_start = oh * pool_size;
        int w_start = ow * pool_size;
        int base_offset = b * (channels * in_height * in_width) + c * (in_height * in_width);
        
        for (int ph = 0; ph < pool_size; ph++) {
            for (int pw = 0; pw < pool_size; pw++) {
                int ih = h_start + ph;
                int iw = w_start + pw;
                if (ih < in_height && iw < in_width) {
                    float val = input[base_offset + ih * in_width + iw] * scale_val;
                    max_val = fmaxf(max_val, val);
                }
            }
        }
        
        max_val = fminf(fmaxf(max_val, clamp_min), clamp_max);
        output[idx] = max_val;
    }
}

torch::Tensor fused_scale_maxpool_clamp_hip(
    torch::Tensor input,
    torch::Tensor scale,
    int pool_size,
    float clamp_min,
    float clamp_max
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_height = in_height / pool_size;
    int out_width = in_width / pool_size;
    
    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());
    
    int total = batch_size * channels * out_height * out_width;
    const int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    
    // Use optimized kernel for pool_size=4 with aligned widths
    if (pool_size == 4 && (out_width * 4) % 4 == 0) {
        fused_scale_maxpool_clamp_kernel_v3<<<num_blocks, block_size>>>(
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
    } else {
        fused_scale_maxpool_clamp_kernel_generic<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            scale.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            channels,
            in_height,
            in_width,
            out_height,
            out_width,
            pool_size,
            clamp_min,
            clamp_max
        );
    }
    
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
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses scale, max pooling, and clamping into a single kernel.
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
        # Fused scale + maxpool + clamp
        x = self.fused_module.fused_scale_maxpool_clamp_hip(
            x, 
            self.scale.view(-1),
            self.maxpool_kernel_size,
            self.clamp_min,
            self.clamp_max
        )
        return x
