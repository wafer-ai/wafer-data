import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized fused kernel
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Optimized tiled kernel for 2x2 pooling with fused operations
// Each thread handles one output element
__global__ void fused_subtract_tanh_subtract_avgpool_opt(
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
    // Calculate global index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = batch_size * channels * out_height * out_width;
    
    if (idx >= total_out) return;
    
    // Decompose linear index into coordinates
    int tmp = idx;
    int ow = tmp % out_width; tmp /= out_width;
    int oh = tmp % out_height; tmp /= out_height;
    int c = tmp % channels; tmp /= channels;
    int b = tmp;
    
    // Input coordinates
    int ih = oh * 2;
    int iw = ow * 2;
    
    // Input stride
    int in_stride_h = in_width;
    int in_stride_c = in_height * in_width;
    int in_stride_b = channels * in_stride_c;
    
    // Base pointer for this batch/channel
    const float* base = input + b * in_stride_b + c * in_stride_c;
    
    // Load 2x2 values - coalesced access pattern for neighboring threads
    float v00 = base[ih * in_stride_h + iw];
    float v01 = base[ih * in_stride_h + iw + 1];
    float v10 = base[(ih + 1) * in_stride_h + iw];
    float v11 = base[(ih + 1) * in_stride_h + iw + 1];
    
    // Fused: subtract1 -> tanh -> subtract2
    v00 = tanhf(v00 - subtract1_value) - subtract2_value;
    v01 = tanhf(v01 - subtract1_value) - subtract2_value;
    v10 = tanhf(v10 - subtract1_value) - subtract2_value;
    v11 = tanhf(v11 - subtract1_value) - subtract2_value;
    
    // Average pooling (2x2)
    output[idx] = (v00 + v01 + v10 + v11) * 0.25f;
}

// Generic version for any pool size
__global__ void fused_subtract_tanh_subtract_avgpool_generic(
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    int tmp = idx;
    int ow = tmp % out_width; tmp /= out_width;
    int oh = tmp % out_height; tmp /= out_height;
    int c = tmp % channels; tmp /= channels;
    int b = tmp;
    
    int ih_start = oh * pool_size;
    int iw_start = ow * pool_size;
    
    int base_offset = b * channels * in_height * in_width + c * in_height * in_width;
    
    float sum = 0.0f;
    float inv_pool = 1.0f / (float)(pool_size * pool_size);
    
    for (int ph = 0; ph < pool_size; ph++) {
        int ih = ih_start + ph;
        for (int pw = 0; pw < pool_size; pw++) {
            int iw = iw_start + pw;
            float val = input[base_offset + ih * in_width + iw];
            val = tanhf(val - subtract1_value) - subtract2_value;
            sum += val;
        }
    }
    
    output[idx] = sum * inv_pool;
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
    
    const int total_elements = batch_size * channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    if (pool_size == 2) {
        fused_subtract_tanh_subtract_avgpool_opt<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, channels, in_height, in_width,
            out_height, out_width,
            subtract1_value, subtract2_value
        );
    } else {
        fused_subtract_tanh_subtract_avgpool_generic<<<num_blocks, block_size>>>(
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
