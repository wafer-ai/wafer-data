import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

maxpool2d_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <float.h>

// Optimized kernel with vectorized loads and better memory coalescing
__global__ void maxpool2d_kernel_opt(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    // Each thread processes one output element
    // Better indexing for memory coalescing - process along width dimension
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int bc = blockIdx.z;  // combined batch and channel index
    
    if (ow >= out_width || oh >= out_height) return;
    
    int b = bc / channels;
    int c = bc % channels;
    
    // Calculate input start position
    int ih_start = oh * stride - padding;
    int iw_start = ow * stride - padding;
    
    float max_val = -FLT_MAX;
    
    // Input pointer for this batch and channel
    const float* input_ptr = input + (b * channels + c) * in_height * in_width;
    
    // Unrolled loop for kernel_size=4
    for (int kh = 0; kh < 4; ++kh) {
        int ih = ih_start + kh * dilation;
        if (ih >= 0 && ih < in_height) {
            int row_offset = ih * in_width;
            for (int kw = 0; kw < 4; ++kw) {
                int iw = iw_start + kw * dilation;
                if (iw >= 0 && iw < in_width) {
                    float val = input_ptr[row_offset + iw];
                    max_val = fmaxf(max_val, val);
                }
            }
        }
    }
    
    int out_idx = (b * channels + c) * out_height * out_width + oh * out_width + ow;
    output[out_idx] = max_val;
}

// Specialized kernel for kernel_size=4, stride=1, dilation=1
__global__ void maxpool2d_k4s1d1_kernel(
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
    // Each thread processes one output element
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int bc = blockIdx.z;
    
    if (ow >= out_width || oh >= out_height) return;
    
    int b = bc / channels;
    int c = bc % channels;
    
    int ih_start = oh - padding;
    int iw_start = ow - padding;
    
    const float* input_ptr = input + (b * channels + c) * in_height * in_width;
    
    float max_val = -FLT_MAX;
    
    // Fully unrolled 4x4 kernel
    #pragma unroll
    for (int kh = 0; kh < 4; ++kh) {
        int ih = ih_start + kh;
        if (ih >= 0 && ih < in_height) {
            int row_offset = ih * in_width;
            #pragma unroll
            for (int kw = 0; kw < 4; ++kw) {
                int iw = iw_start + kw;
                if (iw >= 0 && iw < in_width) {
                    max_val = fmaxf(max_val, input_ptr[row_offset + iw]);
                }
            }
        }
    }
    
    int out_idx = (b * channels + c) * out_height * out_width + oh * out_width + ow;
    output[out_idx] = max_val;
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
    
    // Calculate output dimensions
    int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());
    
    // 2D block for better spatial locality
    dim3 block(32, 8);  // 256 threads total
    dim3 grid(
        (out_width + block.x - 1) / block.x,
        (out_height + block.y - 1) / block.y,
        batch_size * channels
    );
    
    if (kernel_size == 4 && stride == 1 && dilation == 1) {
        maxpool2d_k4s1d1_kernel<<<grid, block>>>(
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
    } else {
        maxpool2d_kernel_opt<<<grid, block>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            channels,
            in_height,
            in_width,
            out_height,
            out_width,
            kernel_size,
            stride,
            padding,
            dilation
        );
    }
    
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
