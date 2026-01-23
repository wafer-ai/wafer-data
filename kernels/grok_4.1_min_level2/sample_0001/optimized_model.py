import os
os.environ["CXX"] = "hipcc"

from torch.utils.cpp_extension import load_inline
import torch
import torch.nn as nn

conv2d_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void tiled_conv2d_kernel(
    const float* input, const float* filter, const float* bias, float* output,
    int batchSize, int inChannels, int inputHeight, int inputWidth,
    int outChannels, int filterHeight, int filterWidth, int outputHeight, int outputWidth,
    int tile_oh_size, int tile_ow_size, int shared_oh_size, int shared_ow_size
) {
    extern __shared__ float shared_data[];
    int shared_pitch = shared_ow_size;
    #define TILE_IN(r, c) (shared_data[(r) * shared_pitch + (c)])

    int nc = blockIdx.z;
    int n = nc / outChannels;
    int cout = nc % outChannels;
    int tile_oh_idx = blockIdx.y;
    int tile_ow_idx = blockIdx.x;
    int oh = tile_oh_idx * tile_oh_size + threadIdx.y;
    int ow = tile_ow_idx * tile_ow_size + threadIdx.x;

    bool is_valid = (oh < outputHeight && ow < outputWidth);
    float accum = 0.0f;
    int ih_start = tile_oh_idx * tile_oh_size;
    int iw_start = tile_ow_idx * tile_ow_size;

    // Load weights for this cout into shared memory
    float* weight_base = shared_data + (shared_oh_size * shared_pitch);
    #define WEIGHT(ci, kh, kw) (weight_base[((ci) * filterHeight + (kh)) * filterWidth + (kw)])
    int num_weight_elements = inChannels * filterHeight * filterWidth;
    for (int widx = threadIdx.y * blockDim.x + threadIdx.x; widx < num_weight_elements; widx += blockDim.x * blockDim.y) {
        int ci_w = widx / (filterHeight * filterWidth);
        int k_local = widx % (filterHeight * filterWidth);
        int kh_w = k_local / filterWidth;
        int kw_w = k_local % filterWidth;
        WEIGHT(ci_w, kh_w, kw_w) = filter[((cout * inChannels + ci_w) * filterHeight + kh_w) * filterWidth + kw_w];
    }
    __syncthreads();

    for (int ci = 0; ci < inChannels; ++ci) {
        // Load input tile to shared memory in phases
        for (int row_phase = 0; row_phase < shared_oh_size; row_phase += blockDim.y) {
            int row = row_phase + threadIdx.y;
            if (row < shared_oh_size) {
                for (int col_phase = 0; col_phase < shared_ow_size; col_phase += blockDim.x) {
                    int col = col_phase + threadIdx.x;
                    if (col < shared_ow_size) {
                        int ih = ih_start + row;
                        int iw = iw_start + col;
                        float val = 0.0f;
                        if (ih < inputHeight && iw < inputWidth) {
                            val = input[((n * inChannels + ci) * inputHeight + ih) * inputWidth + iw];
                        }
                        TILE_IN(row, col) = val;
                    }
                }
            }
        }
        __syncthreads();

        // Compute contribution from this ci only if valid
        if (is_valid) {
            #pragma unroll
            for (int kh = 0; kh < filterHeight; ++kh) {
                int lrow = threadIdx.y + kh;
                if (lrow < shared_oh_size) {
                    #pragma unroll
                    for (int kw = 0; kw < filterWidth; ++kw) {
                        int lcol = threadIdx.x + kw;
                        if (lcol < shared_ow_size) {
                            float i_val = TILE_IN(lrow, lcol);
                            float f_val = WEIGHT(ci, kh, kw);
                            accum += i_val * f_val;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    if (is_valid) {
        accum += bias[cout];
        int out_idx = ((n * outChannels + cout) * outputHeight + oh) * outputWidth + ow;
        output[out_idx] = accum;
    }
}

torch::Tensor conv2d_hip(torch::Tensor input, torch::Tensor filter, torch::Tensor bias) {
    auto batchSize = input.size(0);
    auto inChannels = input.size(1);
    auto inputHeight = input.size(2);
    auto inputWidth = input.size(3);
    auto outChannels = filter.size(0);
    auto filterHeight = filter.size(2);
    auto filterWidth = filter.size(3);
    auto outputHeight = inputHeight - filterHeight + 1;
    auto outputWidth = inputWidth - filterWidth + 1;
    if (outputHeight <= 0 || outputWidth <= 0) {
        return torch::empty({0}, input.options());
    }
    auto output = torch::empty({batchSize, outChannels, outputHeight, outputWidth}, input.options());

    int tile_oh_size = 32;
    int tile_ow_size = 32;
    int halo_h = filterHeight - 1;
    int halo_w = filterWidth - 1;
    int shared_oh_size = tile_oh_size + halo_h;
    int shared_ow_size = tile_ow_size + halo_w;
    size_t input_shared_bytes = (size_t)shared_oh_size * shared_ow_size * sizeof(float);
    size_t weight_shared_bytes = (size_t)inChannels * filterHeight * filterWidth * sizeof(float);
    size_t total_shared_bytes = input_shared_bytes + weight_shared_bytes;

    int num_tile_oh = (outputHeight + tile_oh_size - 1) / tile_oh_size;
    int num_tile_ow = (outputWidth + tile_ow_size - 1) / tile_ow_size;
    dim3 block(tile_oh_size, tile_ow_size);
    dim3 grid(num_tile_ow, num_tile_oh, batchSize * outChannels);

    tiled_conv2d_kernel<<<grid, block, total_shared_bytes>>>(
        input.data_ptr<float>(),
        filter.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batchSize, inChannels, inputHeight, inputWidth,
        outChannels, filterHeight, filterWidth, outputHeight, outputWidth,
        tile_oh_size, tile_ow_size, shared_oh_size, shared_ow_size
    );

    return output;
}
"""

conv2d_ext = load_inline(
    name="conv2d",
    cpp_sources=conv2d_cpp_source,
    functions=["conv2d_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model with custom tiled HIP conv2d kernel with weight tiling, tile=32.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.divide_by = divide_by

    def forward(self, x):
        x = conv2d_ext.conv2d_hip(x, self.conv.weight, self.conv.bias)
        x = self.instance_norm(x)
        x = x / self.divide_by
        return x

batch_size = 128
in_channels  = 64  
out_channels = 128  
height = width = 128  
kernel_size = 3
divide_by = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divide_by]
