import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized fused kernel using vectorized loads
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Vectorized kernel - each thread processes 4 output pixels horizontally
__global__ void fused_subtract_tanh_subtract_avgpool_vec4(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const float sub1,
    const float sub2
) {
    // Each thread processes 4 consecutive output elements along width
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_vec_width = out_width / 4;  // Number of float4 vectors in output width
    int total_vec = batch_size * channels * out_height * out_vec_width;
    
    if (idx >= total_vec) return;
    
    // Decompose index
    int tmp = idx;
    int vec_x = tmp % out_vec_width; tmp /= out_vec_width;
    int oh = tmp % out_height; tmp /= out_height;
    int c = tmp % channels; tmp /= channels;
    int b = tmp;
    
    int ow = vec_x * 4;  // Base output x coordinate
    int ih = oh * 2;     // Input y coordinate
    int iw = ow * 2;     // Input x coordinate
    
    // Strides
    int in_hw = in_height * in_width;
    int out_hw = out_height * out_width;
    
    const float* in_base = input + b * channels * in_hw + c * in_hw;
    
    // Process 4 output pixels = 8x2 input region
    // Row 0 of input
    const float* row0 = in_base + ih * in_width + iw;
    const float* row1 = in_base + (ih + 1) * in_width + iw;
    
    float4 out;
    
    // Output pixel 0: input [0:2, 0:2]
    float v00 = tanhf(row0[0] - sub1) - sub2;
    float v01 = tanhf(row0[1] - sub1) - sub2;
    float v10 = tanhf(row1[0] - sub1) - sub2;
    float v11 = tanhf(row1[1] - sub1) - sub2;
    out.x = (v00 + v01 + v10 + v11) * 0.25f;
    
    // Output pixel 1: input [0:2, 2:4]
    v00 = tanhf(row0[2] - sub1) - sub2;
    v01 = tanhf(row0[3] - sub1) - sub2;
    v10 = tanhf(row1[2] - sub1) - sub2;
    v11 = tanhf(row1[3] - sub1) - sub2;
    out.y = (v00 + v01 + v10 + v11) * 0.25f;
    
    // Output pixel 2: input [0:2, 4:6]
    v00 = tanhf(row0[4] - sub1) - sub2;
    v01 = tanhf(row0[5] - sub1) - sub2;
    v10 = tanhf(row1[4] - sub1) - sub2;
    v11 = tanhf(row1[5] - sub1) - sub2;
    out.z = (v00 + v01 + v10 + v11) * 0.25f;
    
    // Output pixel 3: input [0:2, 6:8]
    v00 = tanhf(row0[6] - sub1) - sub2;
    v01 = tanhf(row0[7] - sub1) - sub2;
    v10 = tanhf(row1[6] - sub1) - sub2;
    v11 = tanhf(row1[7] - sub1) - sub2;
    out.w = (v00 + v01 + v10 + v11) * 0.25f;
    
    // Store result as float4
    float4* out_ptr = (float4*)(output + b * channels * out_hw + c * out_hw + oh * out_width + ow);
    *out_ptr = out;
}

// Scalar version for edge cases
__global__ void fused_subtract_tanh_subtract_avgpool_scalar(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const float sub1,
    const float sub2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_height * out_width;
    
    if (idx >= total) return;
    
    int tmp = idx;
    int ow = tmp % out_width; tmp /= out_width;
    int oh = tmp % out_height; tmp /= out_height;
    int c = tmp % channels; tmp /= channels;
    int b = tmp;
    
    int ih = oh * 2;
    int iw = ow * 2;
    
    int in_hw = in_height * in_width;
    const float* base = input + b * channels * in_hw + c * in_hw;
    
    float v00 = tanhf(base[ih * in_width + iw] - sub1) - sub2;
    float v01 = tanhf(base[ih * in_width + iw + 1] - sub1) - sub2;
    float v10 = tanhf(base[(ih+1) * in_width + iw] - sub1) - sub2;
    float v11 = tanhf(base[(ih+1) * in_width + iw + 1] - sub1) - sub2;
    
    output[idx] = (v00 + v01 + v10 + v11) * 0.25f;
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
    
    const int block_size = 256;
    
    if (pool_size == 2 && out_width % 4 == 0) {
        // Use vectorized kernel
        int out_vec_width = out_width / 4;
        int total_vec = batch_size * channels * out_height * out_vec_width;
        int num_blocks = (total_vec + block_size - 1) / block_size;
        
        fused_subtract_tanh_subtract_avgpool_vec4<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, channels, in_height, in_width,
            out_height, out_width,
            subtract1_value, subtract2_value
        );
    } else {
        // Use scalar kernel
        int total = batch_size * channels * out_height * out_width;
        int num_blocks = (total + block_size - 1) / block_size;
        
        fused_subtract_tanh_subtract_avgpool_scalar<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, channels, in_height, in_width,
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
