import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

conv2d_cpp_source = """
torch::Tensor conv2d_hip(torch::Tensor input, torch::Tensor weight, int stride, int padding);
"""

conv2d_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define TILE_SIZE 16

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding
) {
    // Output position
    int out_x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int out_y = blockIdx.y * TILE_SIZE + threadIdx.y;
    int out_c = blockIdx.z % out_channels;
    int batch = blockIdx.z / out_channels;
    
    if (out_x >= out_width || out_y >= out_height || batch >= batch_size) return;
    
    float sum = 0.0f;
    
    // Input position (top-left corner for this output pixel)
    int in_y_start = out_y * stride - padding;
    int in_x_start = out_x * stride - padding;
    
    // Loop over input channels and kernel
    for (int ic = 0; ic < in_channels; ic++) {
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int in_y = in_y_start + ky;
                int in_x = in_x_start + kx;
                
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    int input_idx = batch * (in_channels * in_height * in_width) +
                                   ic * (in_height * in_width) +
                                   in_y * in_width + in_x;
                    int weight_idx = out_c * (in_channels * kernel_size * kernel_size) +
                                    ic * (kernel_size * kernel_size) +
                                    ky * kernel_size + kx;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    int output_idx = batch * (out_channels * out_height * out_width) +
                    out_c * (out_height * out_width) +
                    out_y * out_width + out_x;
    output[output_idx] = sum;
}

torch::Tensor conv2d_hip(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (out_width + TILE_SIZE - 1) / TILE_SIZE,
        (out_height + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * out_channels
    );
    
    conv2d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding
    );
    
    return output;
}
"""

conv2d_module = load_inline(
    name="conv2d_hip",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_hip_source,
    functions=["conv2d_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Store parameters
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Create weight parameter (same as nn.Conv2d)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights like nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use custom HIP kernel for simple cases
        if self.dilation == 1 and self.groups == 1 and self.bias is None:
            return conv2d_module.conv2d_hip(x, self.weight, self.stride, self.padding)
        else:
            # Fallback to PyTorch for complex cases
            return torch.nn.functional.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def get_inputs():
    x = torch.rand(16, 16, 1024, 1024).cuda()
    return [x]


def get_init_inputs():
    return [16, 128, 3]
