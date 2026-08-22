import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

depthwise_conv_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// MI300X optimized kernel with 256-thread blocks (4 wavefronts)
// Each thread processes 4 outputs along width for ILP
__global__ __launch_bounds__(256, 4)
void depthwise_conv2d_3x3_mi300x(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int padding
) {
    // Block: 64x4 threads, each thread does 4 outputs = 256 outputs per row, 4 rows
    // Tile: 256 x 4 outputs per block
    constexpr int TILE_W = 256;
    constexpr int TILE_H = 4;
    constexpr int OUTPUTS_PER_THREAD = 4;
    constexpr int THREADS_X = 64;
    constexpr int THREADS_Y = 4;
    
    int bc = blockIdx.z;
    int batch = bc / channels;
    int channel = bc % channels;
    
    int tile_x = blockIdx.x * TILE_W;
    int tile_y = blockIdx.y * TILE_H;
    
    int tx = threadIdx.x;  // 0-63
    int ty = threadIdx.y;  // 0-3
    
    // Load weights into registers
    float w0, w1, w2, w3, w4, w5, w6, w7, w8;
    {
        const float* wptr = weight + channel * 9;
        w0 = wptr[0]; w1 = wptr[1]; w2 = wptr[2];
        w3 = wptr[3]; w4 = wptr[4]; w5 = wptr[5];
        w6 = wptr[6]; w7 = wptr[7]; w8 = wptr[8];
    }
    
    const float* input_ch = input + (batch * channels + channel) * in_height * in_width;
    float* output_ch = output + (batch * channels + channel) * out_height * out_width;
    
    // Shared memory: (TILE_H + 2) x (TILE_W + 2)
    constexpr int SMEM_H = TILE_H + 2;
    constexpr int SMEM_W = TILE_W + 2;
    __shared__ float smem[SMEM_H][SMEM_W + 1];  // +1 to reduce bank conflicts
    
    int in_tile_x = tile_x - padding;
    int in_tile_y = tile_y - padding;
    
    // Cooperative loading: 256 threads load (6 * 258) = 1548 elements
    int tid = ty * THREADS_X + tx;
    int nthreads = THREADS_X * THREADS_Y;
    int nelems = SMEM_H * SMEM_W;
    
    for (int i = tid; i < nelems; i += nthreads) {
        int sy = i / SMEM_W;
        int sx = i % SMEM_W;
        int iy = in_tile_y + sy;
        int ix = in_tile_x + sx;
        float v = 0.0f;
        if (iy >= 0 && iy < in_height && ix >= 0 && ix < in_width) {
            v = input_ch[iy * in_width + ix];
        }
        smem[sy][sx] = v;
    }
    
    __syncthreads();
    
    int out_y = tile_y + ty;
    if (out_y >= out_height) return;
    
    int ly = ty;
    
    // Each thread computes 4 consecutive outputs
    #pragma unroll
    for (int k = 0; k < OUTPUTS_PER_THREAD; ++k) {
        int out_x = tile_x + tx * OUTPUTS_PER_THREAD + k;
        if (out_x < out_width) {
            int lx = tx * OUTPUTS_PER_THREAD + k;
            float sum = 0.0f;
            sum += smem[ly][lx] * w0;
            sum += smem[ly][lx+1] * w1;
            sum += smem[ly][lx+2] * w2;
            sum += smem[ly+1][lx] * w3;
            sum += smem[ly+1][lx+1] * w4;
            sum += smem[ly+1][lx+2] * w5;
            sum += smem[ly+2][lx] * w6;
            sum += smem[ly+2][lx+1] * w7;
            sum += smem[ly+2][lx+2] * w8;
            output_ch[out_y * out_width + out_x] = sum;
        }
    }
}

// Version with larger vertical tile (8 rows) for better cache utilization
__global__ __launch_bounds__(256, 4)
void depthwise_conv2d_3x3_tall(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int padding
) {
    // 32x8 threads, each thread does 4 outputs = 128 x 8 tile
    constexpr int TILE_W = 128;
    constexpr int TILE_H = 8;
    constexpr int OUTPUTS_PER_THREAD = 4;
    constexpr int THREADS_X = 32;
    constexpr int THREADS_Y = 8;
    
    int bc = blockIdx.z;
    int batch = bc / channels;
    int channel = bc % channels;
    
    int tile_x = blockIdx.x * TILE_W;
    int tile_y = blockIdx.y * TILE_H;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    float w0, w1, w2, w3, w4, w5, w6, w7, w8;
    {
        const float* wptr = weight + channel * 9;
        w0 = wptr[0]; w1 = wptr[1]; w2 = wptr[2];
        w3 = wptr[3]; w4 = wptr[4]; w5 = wptr[5];
        w6 = wptr[6]; w7 = wptr[7]; w8 = wptr[8];
    }
    
    const float* input_ch = input + (batch * channels + channel) * in_height * in_width;
    float* output_ch = output + (batch * channels + channel) * out_height * out_width;
    
    constexpr int SMEM_H = TILE_H + 2;
    constexpr int SMEM_W = TILE_W + 2;
    __shared__ float smem[SMEM_H][SMEM_W + 1];
    
    int in_tile_x = tile_x - padding;
    int in_tile_y = tile_y - padding;
    
    int tid = ty * THREADS_X + tx;
    int nthreads = THREADS_X * THREADS_Y;
    int nelems = SMEM_H * SMEM_W;
    
    for (int i = tid; i < nelems; i += nthreads) {
        int sy = i / SMEM_W;
        int sx = i % SMEM_W;
        int iy = in_tile_y + sy;
        int ix = in_tile_x + sx;
        float v = 0.0f;
        if (iy >= 0 && iy < in_height && ix >= 0 && ix < in_width) {
            v = input_ch[iy * in_width + ix];
        }
        smem[sy][sx] = v;
    }
    
    __syncthreads();
    
    int out_y = tile_y + ty;
    if (out_y >= out_height) return;
    
    int ly = ty;
    
    #pragma unroll
    for (int k = 0; k < OUTPUTS_PER_THREAD; ++k) {
        int out_x = tile_x + tx * OUTPUTS_PER_THREAD + k;
        if (out_x < out_width) {
            int lx = tx * OUTPUTS_PER_THREAD + k;
            float sum = smem[ly][lx] * w0
                      + smem[ly][lx+1] * w1
                      + smem[ly][lx+2] * w2
                      + smem[ly+1][lx] * w3
                      + smem[ly+1][lx+1] * w4
                      + smem[ly+1][lx+2] * w5
                      + smem[ly+2][lx] * w6
                      + smem[ly+2][lx+1] * w7
                      + smem[ly+2][lx+2] * w8;
            output_ch[out_y * out_width + out_x] = sum;
        }
    }
}

__global__ void depthwise_conv2d_generic(
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
    
    const float* w_ptr = weight + channel * kernel_size * kernel_size;
    const float* in_ptr = input + (batch * channels + channel) * in_height * in_width;
    
    for (int ky = 0; ky < kernel_size; ++ky) {
        int in_y = in_y_start + ky;
        if (in_y >= 0 && in_y < in_height) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int in_x = in_x_start + kx;
                if (in_x >= 0 && in_x < in_width) {
                    sum += in_ptr[in_y * in_width + in_x] * w_ptr[ky * kernel_size + kx];
                }
            }
        }
    }
    
    output[(batch * channels + channel) * out_height * out_width + out_y * out_width + out_x] = sum;
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
        dim3 block(32, 8);  // 256 threads
        dim3 grid(
            (out_width + TILE_W - 1) / TILE_W,
            (out_height + TILE_H - 1) / TILE_H,
            batch_size * channels
        );
        depthwise_conv2d_3x3_tall<<<grid, block>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, channels,
            in_height, in_width,
            out_height, out_width,
            padding
        );
    } else {
        dim3 block(16, 16);
        dim3 grid(
            (out_width + 15) / 16,
            (out_height + 15) / 16,
            batch_size * channels
        );
        depthwise_conv2d_generic<<<grid, block>>>(
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
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_flag = bias
        
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=2.236)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
