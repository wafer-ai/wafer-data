import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

depthwise_conv_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Each thread processes multiple output elements along the width dimension
// for better memory throughput and instruction level parallelism
__global__ void depthwise_conv2d_3x3_vec4(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int stride,
    int padding
) {
    constexpr int TILE_W = 128;  // Width tile
    constexpr int TILE_H = 8;    // Height tile
    constexpr int ITEMS_PER_THREAD = 4;
    
    // 32 threads per row, each processes 4 elements = 128 elements per row
    // 8 rows per block
    constexpr int THREADS_PER_ROW = 32;
    
    int bc = blockIdx.z;
    int batch = bc / channels;
    int channel = bc % channels;
    
    int tile_x = blockIdx.x * TILE_W;
    int tile_y = blockIdx.y * TILE_H;
    
    int tx = threadIdx.x;  // 0-31
    int ty = threadIdx.y;  // 0-7
    
    int out_y = tile_y + ty;
    
    // Load weights into registers
    float w0, w1, w2, w3, w4, w5, w6, w7, w8;
    const float* weight_ptr = weight + channel * 9;
    w0 = weight_ptr[0]; w1 = weight_ptr[1]; w2 = weight_ptr[2];
    w3 = weight_ptr[3]; w4 = weight_ptr[4]; w5 = weight_ptr[5];
    w6 = weight_ptr[6]; w7 = weight_ptr[7]; w8 = weight_ptr[8];
    
    const float* input_base = input + (batch * channels + channel) * in_height * in_width;
    float* output_base = output + (batch * channels + channel) * out_height * out_width;
    
    // Shared memory for input tile
    // Need TILE_W + 2 columns (for 3x3 kernel halo)
    // Need TILE_H + 2 rows
    constexpr int smem_w = TILE_W + 2;
    constexpr int smem_h = TILE_H + 2;
    __shared__ float smem_input[smem_h][smem_w + 1];  // +1 to avoid bank conflicts
    
    int in_tile_x = tile_x * stride - padding;
    int in_tile_y = tile_y * stride - padding;
    
    // Cooperative loading - each thread loads multiple elements
    int thread_id = ty * THREADS_PER_ROW + tx;
    int total_threads = THREADS_PER_ROW * TILE_H;
    int total_elements = smem_h * smem_w;
    
    for (int idx = thread_id; idx < total_elements; idx += total_threads) {
        int sy = idx / smem_w;
        int sx = idx % smem_w;
        
        int in_y = in_tile_y + sy;
        int in_x = in_tile_x + sx;
        
        float val = 0.0f;
        if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
            val = input_base[in_y * in_width + in_x];
        }
        smem_input[sy][sx] = val;
    }
    
    __syncthreads();
    
    if (out_y < out_height) {
        int ly = ty * stride;
        
        // Each thread processes ITEMS_PER_THREAD consecutive output elements
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            int out_x = tile_x + tx * ITEMS_PER_THREAD + i;
            if (out_x < out_width) {
                int lx = (tx * ITEMS_PER_THREAD + i) * stride;
                
                float sum = smem_input[ly][lx] * w0
                          + smem_input[ly][lx+1] * w1
                          + smem_input[ly][lx+2] * w2
                          + smem_input[ly+1][lx] * w3
                          + smem_input[ly+1][lx+1] * w4
                          + smem_input[ly+1][lx+2] * w5
                          + smem_input[ly+2][lx] * w6
                          + smem_input[ly+2][lx+1] * w7
                          + smem_input[ly+2][lx+2] * w8;
                
                output_base[out_y * out_width + out_x] = sum;
            }
        }
    }
}

// Fallback kernel for other kernel sizes
__global__ void depthwise_conv2d_kernel_generic(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int bc = blockIdx.z;
    int batch = bc / channels;
    int channel = bc % channels;
    
    if (out_x >= out_width || out_y >= out_height) return;
    
    float sum = 0.0f;
    
    int in_y_start = out_y * stride - padding;
    int in_x_start = out_x * stride - padding;
    
    const float* weight_ptr = weight + channel * kernel_size * kernel_size;
    const float* input_ptr = input + (batch * channels + channel) * in_height * in_width;
    
    for (int ky = 0; ky < kernel_size; ++ky) {
        int in_y = in_y_start + ky;
        if (in_y >= 0 && in_y < in_height) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int in_x = in_x_start + kx;
                if (in_x >= 0 && in_x < in_width) {
                    sum += input_ptr[in_y * in_width + in_x] * weight_ptr[ky * kernel_size + kx];
                }
            }
        }
    }
    
    int out_idx = ((batch * channels + channel) * out_height + out_y) * out_width + out_x;
    output[out_idx] = sum;
}

torch::Tensor depthwise_conv2d_hip(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int kernel_size = weight.size(2);
    
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());
    
    if (kernel_size == 3 && stride == 1) {
        constexpr int TILE_W = 128;
        constexpr int TILE_H = 8;
        dim3 block(32, 8);  // 256 threads per block
        dim3 grid(
            (out_width + TILE_W - 1) / TILE_W,
            (out_height + TILE_H - 1) / TILE_H,
            batch_size * channels
        );
        depthwise_conv2d_3x3_vec4<<<grid, block>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, channels,
            in_height, in_width,
            out_height, out_width,
            stride, padding
        );
    } else {
        dim3 block(16, 16);
        dim3 grid(
            (out_width + 15) / 16,
            (out_height + 15) / 16,
            batch_size * channels
        );
        depthwise_conv2d_kernel_generic<<<grid, block>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, channels,
            in_height, in_width,
            out_height, out_width,
            kernel_size, stride, padding
        );
    }
    
    return output;
}
"""

cpp_source = """
torch::Tensor depthwise_conv2d_hip(torch::Tensor input, torch::Tensor weight, int stride, int padding);
"""

depthwise_conv_module = load_inline(
    name="depthwise_conv2d",
    cpp_sources=cpp_source,
    cuda_sources=depthwise_conv_source,
    functions=["depthwise_conv2d_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution using custom HIP kernel.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_flag = bias
        
        # Initialize weights the same way PyTorch does
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=2.236)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape weight from (C, 1, K, K) to (C, K, K) for our kernel
        weight_reshaped = self.weight.squeeze(1)
        
        output = depthwise_conv_module.depthwise_conv2d_hip(
            x.contiguous(),
            weight_reshaped.contiguous(),
            self.stride,
            self.padding
        )
        
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        
        return output
