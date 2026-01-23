
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

def get_max_pool2d_lib(kernel_size, stride, padding, dilation):
    max_pool2d_cpp_source = f"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <float.h>

#define TILE_H 16
#define TILE_W 16
#define KERNEL_SIZE {kernel_size}
#define STRIDE {stride}
#define PADDING {padding}
#define DILATION {dilation}

__global__ void __launch_bounds__(256) max_pool2d_shm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int H_out, int W_out) {{

    constexpr int tile_h_in = TILE_H * STRIDE + (KERNEL_SIZE - 1) * DILATION;
    constexpr int tile_w_in = TILE_W * STRIDE + (KERNEL_SIZE - 1) * DILATION;
    __shared__ float shm_input[tile_h_in * tile_w_in];

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int tid = ty * TILE_W + tx;

    int w_out_start = blockIdx.x * TILE_W;
    int h_out_start = blockIdx.y * TILE_H;
    int nc = blockIdx.z;
    int n = nc / C;
    int c = nc % C;

    const float* input_ptr = input + (n * C + c) * (H * W);

    // Coalesced loading into shared memory
    #pragma unroll
    for (int i = tid; i < tile_h_in * tile_w_in; i += TILE_H * TILE_W) {{
        int i_h = i / tile_w_in;
        int i_w = i % tile_w_in;
        int h_in = h_out_start * STRIDE - PADDING + i_h;
        int w_in = w_out_start * STRIDE - PADDING + i_w;
        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {{
            shm_input[i] = input_ptr[h_in * W + w_in];
        }} else {{
            shm_input[i] = -FLT_MAX;
        }}
    }}

    __syncthreads();

    int h_out = h_out_start + ty;
    int w_out = w_out_start + tx;

    if (h_out < H_out && w_out < W_out) {{
        float max_val = -FLT_MAX;
        #pragma unroll
        for (int kh = 0; kh < KERNEL_SIZE; ++kh) {{
            #pragma unroll
            for (int kw = 0; kw < KERNEL_SIZE; ++kw) {{
                float val = shm_input[(ty * STRIDE + kh * DILATION) * tile_w_in + (tx * STRIDE + kw * DILATION)];
                if (val > max_val) {{
                    max_val = val;
                }}
            }}
        }}
        output[((n * C + c) * H_out + h_out) * W_out + w_out] = max_val;
    }}
}}

torch::Tensor max_pool2d_hip(
    torch::Tensor input,
    int kernel_size, int stride, int padding, int dilation) {{

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    int H_out = (H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int W_out = (W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({{N, C, H_out, W_out}}, input.options());

    dim3 block(TILE_W, TILE_H);
    dim3 grid((W_out + TILE_W - 1) / TILE_W, (H_out + TILE_H - 1) / TILE_H, N * C);

    max_pool2d_shm_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        H_out, W_out);

    return output;
}}
"""
    return load_inline(
        name=f"max_pool2d_lib_{kernel_size}_{stride}_{padding}_{dilation}",
        cpp_sources=max_pool2d_cpp_source,
        functions=["max_pool2d_hip"],
        verbose=True,
    )

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.max_pool2d_lib = get_max_pool2d_lib(kernel_size, stride, padding, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_pool2d_lib.max_pool2d_hip(
            x, self.kernel_size, self.stride, self.padding, self.dilation
        )

def get_inputs():
    batch_size = 32
    channels = 64
    height = 512
    width = 512
    x = torch.rand(batch_size, channels, height, width).cuda()
    return [x]

def get_init_inputs():
    kernel_size = 4
    stride = 1
    padding = 1
    dilation = 1
    return [kernel_size, stride, padding, dilation]
