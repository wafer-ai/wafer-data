import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized kernel with vectorized loads
fused_kernel_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Process one row of output per thread block
// Each thread handles multiple output elements
__global__ __launch_bounds__(256)
void fused_subtract_tanh_subtract_avgpool_row(
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
    // blockIdx.x = bc (batch * channel)
    // blockIdx.y = output row
    // threadIdx.x = output column / 4 (vec4)
    
    const int bc = blockIdx.x;
    const int oh = blockIdx.y;
    
    if (oh >= out_height) return;
    
    const int in_hw = in_height * in_width;
    const int out_hw = out_height * out_width;
    
    const int ih = oh * 2;
    
    // Input base for this channel/batch
    const float* in_row0 = input + bc * in_hw + ih * in_width;
    const float* in_row1 = in_row0 + in_width;
    
    // Output base
    float* out_row = output + bc * out_hw + oh * out_width;
    
    // Each thread processes multiple output elements
    for (int ow = threadIdx.x; ow < out_width; ow += blockDim.x) {
        int iw = ow * 2;
        
        float v00 = in_row0[iw];
        float v01 = in_row0[iw + 1];
        float v10 = in_row1[iw];
        float v11 = in_row1[iw + 1];
        
        v00 = tanhf(v00 - sub1) - sub2;
        v01 = tanhf(v01 - sub1) - sub2;
        v10 = tanhf(v10 - sub1) - sub2;
        v11 = tanhf(v11 - sub1) - sub2;
        
        out_row[ow] = (v00 + v01 + v10 + v11) * 0.25f;
    }
}

// Optimized kernel using float2 loads for coalesced access
__global__ __launch_bounds__(256)
void fused_subtract_tanh_subtract_avgpool_coalesced(
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
    // Global output element index
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_channels * out_height * out_width;
    
    if (gid >= total) return;
    
    // Decompose
    int bc = gid / (out_height * out_width);
    int rem = gid % (out_height * out_width);
    int oh = rem / out_width;
    int ow = rem % out_width;
    
    int ih = oh * 2;
    int iw = ow * 2;
    
    const int in_hw = in_height * in_width;
    const float* base = input + bc * in_hw;
    
    // Use float2 loads for consecutive elements
    float2 row0 = *reinterpret_cast<const float2*>(base + ih * in_width + iw);
    float2 row1 = *reinterpret_cast<const float2*>(base + (ih + 1) * in_width + iw);
    
    float v00 = tanhf(row0.x - sub1) - sub2;
    float v01 = tanhf(row0.y - sub1) - sub2;
    float v10 = tanhf(row1.x - sub1) - sub2;
    float v11 = tanhf(row1.y - sub1) - sub2;
    
    output[gid] = (v00 + v01 + v10 + v11) * 0.25f;
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
    
    if (pool_size == 2) {
        // Try coalesced kernel
        const int total = batch_channels * out_height * out_width;
        const int block_size = 256;
        const int num_blocks = (total + block_size - 1) / block_size;
        
        fused_subtract_tanh_subtract_avgpool_coalesced<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_channels, in_height, in_width,
            out_height, out_width,
            subtract1_value, subtract2_value
        );
    } else {
        // Row-based kernel for other cases
        dim3 grid(batch_channels, out_height);
        
        fused_subtract_tanh_subtract_avgpool_row<<<grid, 256>>>(
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
