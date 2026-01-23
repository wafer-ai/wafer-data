import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

maxpool2d_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <float.h>

// Optimized Max Pooling 2D kernel
__global__ void maxpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    // Calculate output position
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int bc = blockIdx.z;  // Combined batch and channel index
    
    if (out_x >= output_width || out_y >= output_height)
        return;
    
    const int b = bc / channels;
    const int c = bc % channels;
    
    // Calculate input start position
    const int in_y_start = out_y * stride - padding;
    const int in_x_start = out_x * stride - padding;
    
    float max_val = -FLT_MAX;
    
    // Iterate over the pooling window
    #pragma unroll
    for (int ky = 0; ky < kernel_size; ++ky) {
        const int in_y = in_y_start + ky * dilation;
        if (in_y >= 0 && in_y < input_height) {
            #pragma unroll
            for (int kx = 0; kx < kernel_size; ++kx) {
                const int in_x = in_x_start + kx * dilation;
                if (in_x >= 0 && in_x < input_width) {
                    const int input_idx = ((b * channels + c) * input_height + in_y) * input_width + in_x;
                    float val = input[input_idx];
                    max_val = fmaxf(max_val, val);
                }
            }
        }
    }
    
    const int output_idx = ((b * channels + c) * output_height + out_y) * output_width + out_x;
    output[output_idx] = max_val;
}

torch::Tensor maxpool2d_hip(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    
    // Calculate output dimensions
    const int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());
    
    // Use 2D thread blocks for better spatial locality
    dim3 block(16, 16);
    dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y,
        batch_size * channels
    );
    
    maxpool2d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        dilation
    );
    
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
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 2D using custom HIP kernel.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return maxpool2d_module.maxpool2d_hip(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation
        )


batch_size = 32
channels = 64
height = 512
width = 512
kernel_size = 4
stride = 1
padding = 1
dilation = 1

def get_inputs():
    x = torch.rand(batch_size, channels, height, width).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]
