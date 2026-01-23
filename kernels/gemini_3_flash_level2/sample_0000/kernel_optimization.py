
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

fused_all_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>
#include <float.h>

__global__ void fused_softmax_maxpool_final_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D, int H, int W,
    int D2, int H2, int W2,
    int K) {

    int n = blockIdx.y;
    int out_spatial_idx = blockIdx.x;
    if (out_spatial_idx >= D2 * H2 * W2) return;

    int d2 = out_spatial_idx / (H2 * W2);
    int h2 = (out_spatial_idx / W2) % H2;
    int w2 = out_spatial_idx % W2;

    int K2 = K * K;
    int t = threadIdx.x;

    // Use a larger buffer for shared memory to handle more channels if needed
    // 32 channels, 64 pixels = 32 * 64 * 4 = 8192 bytes
    __shared__ float shared_softmax[32][64];

    int i0 = t / 16;
    int j0 = (t / 4) % 4;
    int k0 = t % 4;

    int d = d2 * K2 + i0;
    int h = h2 * K2 + j0;
    int w = w2 * K2 + k0;

    if (d < D && h < H && w < W) {
        int base_idx = ((n * C) * D + d) * H + h;
        int stride = D * H * W;

        float max_val = -FLT_MAX;
        float vals[32]; // Fixed size
        for (int c = 0; c < C && c < 32; ++c) {
            float v = input[base_idx + c * stride + w];
            vals[c] = v;
            if (v > max_val) max_val = v;
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < C && c < 32; ++c) {
            float e = expf(vals[c] - max_val);
            vals[c] = e;
            sum_exp += e;
        }

        float inv_sum_exp = 1.0f / sum_exp;
        for (int c = 0; c < C && c < 32; ++c) {
            shared_softmax[c][t] = vals[c] * inv_sum_exp;
        }
    } else {
        for (int c = 0; c < C && c < 32; ++c) {
            shared_softmax[c][t] = -FLT_MAX;
        }
    }

    __syncthreads();

    if (t < C && t < 32) {
        float res = -FLT_MAX;
        for (int p = 0; p < 64; ++p) {
            if (shared_softmax[t][p] > res) res = shared_softmax[t][p];
        }
        output[(((n * C + t) * D2 + d2) * H2 + h2) * W2 + w2] = res;
    }
}

torch::Tensor fused_softmax_maxpool_final_hip(torch::Tensor input, int K, int D2, int H2, int W2) {
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    auto output = torch::empty({N, C, D2, H2, W2}, input.options());
    dim3 block(64);
    dim3 grid(D2 * H2 * W2, N);
    fused_softmax_maxpool_final_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), N, C, D, H, W, D2, H2, W2, K);
    return output;
}
"""

fused_all = load_inline(
    name="fused_all",
    cpp_sources=fused_all_source,
    functions=["fused_softmax_maxpool_final_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size
        self.fused_all = fused_all

    def forward(self, x):
        x = self.conv(x)
        N, C, D, H, W = x.shape
        K = self.pool_kernel_size
        D1, H1, W1 = D // K, H // K, W // K
        D2, H2, W2 = D1 // K, H1 // K, W1 // K
        
        if K == 2 and C <= 32:
            x = self.fused_all.fused_softmax_maxpool_final_hip(x, K, D2, H2, W2)
        else:
            x = torch.softmax(x, dim=1)
            x = F.max_pool3d(x, K)
            x = F.max_pool3d(x, K)
        return x

def get_inputs():
    batch_size = 128
    in_channels = 3
    depth, height, width = 16, 32, 32
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    pool_kernel_size = 2
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
