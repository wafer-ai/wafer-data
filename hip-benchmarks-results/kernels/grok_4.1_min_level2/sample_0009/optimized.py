import os
os.environ["CXX"] = "hipcc"
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define TILE_H 16
#define TILE_W 16
#define K 3
#define IN_TILE_H (TILE_H + K - 1)
#define IN_TILE_W (TILE_W + K - 1)
#define MAX_CIN 8

__shared__ float s_input[IN_TILE_H][IN_TILE_W][MAX_CIN];

__global__ void conv2d_tiled_kernel(const float* input, const float* weight, const float* bias, float* output,
                                    const int N, const int Cin, const int Cout, const int Hin, const int Win,
                                    const int out_h, const int out_w, const int num_oh_tiles, const int num_ow_tiles) {
    int tile_id = blockIdx.x;
    int sub_tile_id = tile_id % (num_oh_tiles * num_ow_tiles);
    int img_id = tile_id / (num_oh_tiles * num_ow_tiles);
    int b = img_id / Cout;
    int cout = img_id % Cout;
    int tile_oh_id = sub_tile_id / num_ow_tiles;
    int tile_ow_id = sub_tile_id % num_ow_tiles;

    int oh_start = tile_oh_id * TILE_H;
    int ow_start = tile_ow_id * TILE_W;
    int ih_start = oh_start;
    int iw_start = ow_start;

    // Load input tile to shared memory - channel-wise spatial linear coalesced loads
    int tid = threadIdx.x;
    int num_spatial = IN_TILE_H * IN_TILE_W;
#pragma unroll
    for (int lc = 0; lc < Cin; ++lc) {
        int num_loads = (num_spatial + 255) / 256;
        for (int load_id = 0; load_id < num_loads; ++load_id) {
            int elem_id = tid + load_id * 256;
            if (elem_id < num_spatial) {
                int lih = elem_id / IN_TILE_W;
                int liw = elem_id % IN_TILE_W;
                int ih = ih_start + lih;
                int iw = iw_start + liw;
                float val = 0.0f;
                if (ih < Hin && iw < Win) {
                    val = input[((b * Cin + lc) * Hin + ih) * Win + iw];
                }
                s_input[lih][liw][lc] = val;
            }
        }
    }
    __syncthreads();

    // Compute
    int ty = tid / TILE_W;
    int tx = tid % TILE_W;
    int oh = oh_start + ty;
    int ow = ow_start + tx;
    if (oh < out_h && ow < out_w) {
        float sum = bias[cout];
#pragma unroll
        for (int cin = 0; cin < Cin; ++cin) {
#pragma unroll
            for (int ky = 0; ky < K; ++ky) {
                int sih = ty + ky;
#pragma unroll
                for (int kx = 0; kx < K; ++kx) {
                    int siw = tx + kx;
                    float w_val = weight[((cout * Cin + cin) * K + ky) * K + kx];
                    sum += s_input[sih][siw][cin] * w_val;
                }
            }
        }
        int out_idx = ((b * Cout + cout) * out_h + oh) * out_w + ow;
        output[out_idx] = sum;
    }
}

torch::Tensor conv2d_bn_scale_hip(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto N = input.size(0);
    auto Cin = input.size(1);
    auto Hin = input.size(2);
    auto Win = input.size(3);
    auto Cout = weight.size(0);
    auto Kh = weight.size(2);
    auto Kw = weight.size(3);
    if (Kh != K || Kw != K || Cin > MAX_CIN) {
        throw std::runtime_error("Unsupported shape");
    }
    auto out_h_i64 = Hin - K + 1;
    auto out_w_i64 = Win - K + 1;
    int out_h = static_cast<int>(out_h_i64);
    int out_w = static_cast<int>(out_w_i64);
    auto opts = input.options();
    auto output = torch::empty({N, Cout, out_h_i64, out_w_i64}, opts);

    int64_t tile_h = TILE_H;
    int64_t tile_w = TILE_W;
    int64_t num_oh_tiles = (static_cast<int64_t>(out_h) + tile_h - 1) / tile_h;
    int64_t num_ow_tiles = (static_cast<int64_t>(out_w) + tile_w - 1) / tile_w;
    int64_t num_blocks_i64 = N * Cout * num_oh_tiles * num_ow_tiles;
    dim3 grid(static_cast<unsigned int>(num_blocks_i64));
    dim3 block(256);
    hipLaunchKernelGGL(conv2d_tiled_kernel, grid, block, 0, 0,
                       input.data_ptr<float>(),
                       weight.data_ptr<float>(),
                       bias.data_ptr<float>(),
                       output.data_ptr<float>(),
                       static_cast<int>(N), static_cast<int>(Cin), static_cast<int>(Cout),
                       static_cast<int>(Hin), static_cast<int>(Win),
                       out_h, out_w,
                       static_cast<int>(num_oh_tiles), static_cast<int>(num_ow_tiles));
    return output;
}
"""

conv_hip_module = load_inline(
    name="fused_conv_bn",
    cpp_sources=cpp_source,
    functions=["conv2d_bn_scale_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor
        self.conv_hip = conv_hip_module

    def forward(self, x):
        eps = self.bn.eps
        denom = torch.sqrt(self.bn.running_var + eps)
        eff_gamma = self.bn.weight / denom * self.scaling_factor
        conv_bias = self.conv.bias
        eff_bias = ((conv_bias - self.bn.running_mean) * eff_gamma + self.bn.bias * self.scaling_factor)
        eff_weight = self.conv.weight * eff_gamma.view(self.conv.out_channels, 1, 1, 1)
        out = self.conv_hip.conv2d_bn_scale_hip(x, eff_weight, eff_bias)
        return out

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor]
