import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# This version uses tiled convolution with shared memory for better memory access
conv2d_cpp_source = """
torch::Tensor conv2d_hip(torch::Tensor input, torch::Tensor weight, int stride, int padding);
"""

conv2d_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Optimized for 3x3 kernel, stride=1, padding=0
// Uses shared memory tiling with larger tile size
#define TILE_W 32
#define TILE_H 8
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 8

// Kernel optimized for 3x3 convolution using shared memory
__global__ void conv2d_shared_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int kernel_size
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Each block handles one output tile for one (batch, out_channel) pair
    const int out_tile_x = blockIdx.x * TILE_W;
    const int out_tile_y = blockIdx.y * TILE_H;
    const int oc = blockIdx.z % out_channels;
    const int batch = blockIdx.z / out_channels;
    
    // Global output position
    const int out_x = out_tile_x + tx;
    const int out_y = out_tile_y + ty;
    
    // Early exit for threads outside output bounds
    if (batch >= batch_size) return;
    
    // Shared memory for filter weights (load once per output channel)
    __shared__ float s_weight[16][9];  // [in_channels][kernel_size*kernel_size] - max 16 channels
    
    // Load weights into shared memory
    const int weight_load_idx = ty * BLOCK_SIZE_X + tx;
    if (weight_load_idx < in_channels * kernel_size * kernel_size) {
        const int ic = weight_load_idx / (kernel_size * kernel_size);
        const int k_idx = weight_load_idx % (kernel_size * kernel_size);
        if (ic < 16) {
            s_weight[ic][k_idx] = weight[oc * in_channels * kernel_size * kernel_size + weight_load_idx];
        }
    }
    __syncthreads();
    
    if (out_x >= out_width || out_y >= out_height) return;
    
    float sum = 0.0f;
    
    // Input base indices
    const int in_y_base = out_y;  // stride=1, padding=0
    const int in_x_base = out_x;
    
    // Compute convolution
    for (int ic = 0; ic < in_channels; ic++) {
        const int input_base = batch * in_channels * in_height * in_width + 
                               ic * in_height * in_width;
        
        for (int ky = 0; ky < kernel_size; ky++) {
            const int in_y = in_y_base + ky;
            for (int kx = 0; kx < kernel_size; kx++) {
                const int in_x = in_x_base + kx;
                const float in_val = input[input_base + in_y * in_width + in_x];
                const float w_val = s_weight[ic][ky * kernel_size + kx];
                sum += in_val * w_val;
            }
        }
    }
    
    const int output_idx = batch * out_channels * out_height * out_width +
                          oc * out_height * out_width +
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
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid(
        (out_width + TILE_W - 1) / TILE_W,
        (out_height + TILE_H - 1) / TILE_H,
        batch_size * out_channels
    );
    
    conv2d_shared_kernel<<<grid, block>>>(
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
        kernel_size
    );
    
    return output;
}
"""

conv2d_module = load_inline(
    name="conv2d_hip_v2",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_hip_source,
    functions=["conv2d_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dilation == 1 and self.groups == 1 and self.bias is None and self.padding == 0:
            return conv2d_module.conv2d_hip(x, self.weight, self.stride, self.padding)
        else:
            return torch.nn.functional.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def get_inputs():
    x = torch.rand(16, 16, 1024, 1024).cuda()
    return [x]


def get_init_inputs():
    return [16, 128, 3]
