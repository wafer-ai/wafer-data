
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

maxpool2d_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <algorithm>
#include <float.h>

#define TILE_W 16
#define TILE_H 16

__global__ void __launch_bounds__(256) maxpool2d_kernel_shm_v2(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int k_h, int k_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int H_out, int W_out)
{
    // Increased shared memory size to handle more general cases.
    __shared__ float shm_tile[48][48];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int nc_idx = blockIdx.z;

    int w_out_start = bx * TILE_W;
    int h_out_start = by * TILE_H;

    int h_in_start = h_out_start * stride_h - padding_h;
    int w_in_start = w_out_start * stride_w - padding_w;

    int input_tile_h = TILE_H * stride_h + (k_h - 1) * dilation_h;
    int input_tile_w = TILE_W * stride_w + (k_w - 1) * dilation_w;
    
    // Safety check for shared memory size
    if (input_tile_h > 48) input_tile_h = 48;
    if (input_tile_w > 48) input_tile_w = 48;

    const float* input_ptr = input + (nc_idx * H * W);

    // Optimized loading into shared memory
    int total_threads = TILE_W * TILE_H;
    int thread_id = ty * TILE_W + tx;
    int total_elements = input_tile_h * input_tile_w;

    for (int idx = thread_id; idx < total_elements; idx += total_threads) {
        int i = idx / input_tile_w;
        int j = idx % input_tile_w;
        int h_in = h_in_start + i;
        int w_in = w_in_start + j;
        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            shm_tile[i][j] = input_ptr[h_in * W + w_in];
        } else {
            shm_tile[i][j] = -FLT_MAX;
        }
    }
    __syncthreads();

    int w_out_idx = w_out_start + tx;
    int h_out_idx = h_out_start + ty;

    if (w_out_idx < W_out && h_out_idx < H_out) {
        float max_val = -FLT_MAX;
        int h_shm_base = ty * stride_h;
        int w_shm_base = tx * stride_w;

        for (int kh = 0; kh < k_h; ++kh) {
            int h_shm = h_shm_base + kh * dilation_h;
            for (int kw = 0; kw < k_w; ++kw) {
                int w_shm = w_shm_base + kw * dilation_w;
                float val = shm_tile[h_shm][w_shm];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
        output[(nc_idx * H_out + h_out_idx) * W_out + w_out_idx] = max_val;
    }
}

torch::Tensor maxpool2d_hip(
    torch::Tensor input,
    int kernel_size, int stride, int padding, int dilation) {
    
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    int k_h = kernel_size;
    int k_w = kernel_size;
    int stride_h = stride;
    int stride_w = stride;
    int padding_h = padding;
    int padding_w = padding;
    int dilation_h = dilation;
    int dilation_w = dilation;

    int H_out = (H + 2 * padding_h - dilation_h * (k_h - 1) - 1) / stride_h + 1;
    int W_out = (W + 2 * padding_w - dilation_w * (k_w - 1) - 1) / stride_w + 1;

    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    dim3 block_size(TILE_W, TILE_H);
    dim3 num_blocks((W_out + TILE_W - 1) / TILE_W,
                    (H_out + TILE_H - 1) / TILE_H,
                    N * C);

    maxpool2d_kernel_shm_v2<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        k_h, k_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        H_out, W_out
    );

    return output;
}
"""

maxpool2d_lib = load_inline(
    name="maxpool2d_shm_v2",
    cpp_sources=maxpool2d_cpp_source,
    functions=["maxpool2d_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.maxpool2d_lib = maxpool2d_lib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool2d_lib.maxpool2d_hip(x, self.kernel_size, self.stride, self.padding, self.dilation)
