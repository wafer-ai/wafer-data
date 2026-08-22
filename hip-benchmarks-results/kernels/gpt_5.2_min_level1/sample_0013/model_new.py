import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Force hipcc
os.environ.setdefault("CXX", "hipcc")

# Depthwise 3x3 stride1 padding0 FP32 NCHW
# Optimizations: shared-memory input tile, unrolled 3x3, one (n,c) per block, 16x16 output tile.

src = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA/HIP tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")

// blockDim = (TILE_W, TILE_H, 1)
// grid = (ceil_div(Wout, TILE_W), ceil_div(Hout, TILE_H), N*C)

template<int TILE_W, int TILE_H>
__global__ void dwconv3x3_s1p0_fwd_kernel(const float* __restrict__ x,
                                        const float* __restrict__ w,
                                        float* __restrict__ y,
                                        int N, int C, int H, int W) {
    // Output dimensions
    const int Hout = H - 2;
    const int Wout = W - 2;

    const int nc = (int)blockIdx.z;
    const int n = nc / C;
    const int c = nc - n * C;

    const int ox0 = (int)blockIdx.x * TILE_W;
    const int oy0 = (int)blockIdx.y * TILE_H;

    const int tx = (int)threadIdx.x; // [0, TILE_W)
    const int ty = (int)threadIdx.y; // [0, TILE_H)

    // Shared tile with 1-pixel halo for 3x3
    __shared__ float sh[(TILE_H + 2) * (TILE_W + 2)];

    // Global input base for this (n,c)
    const int in_base = ((n * C + c) * H) * W;

    // Cooperative load: each thread loads multiple elements
    // Shared dimensions
    const int shW = TILE_W + 2;
    const int shH = TILE_H + 2;

    // Load tile covering input region [oy0, oy0+TILE_H+1] x [ox0, ox0+TILE_W+1]
    for (int l = ty * TILE_W + tx; l < shW * shH; l += TILE_W * TILE_H) {
        int sy = l / shW;
        int sx = l - sy * shW;
        int iy = oy0 + sy;
        int ix = ox0 + sx;
        float v = 0.0f;
        if (iy < H && ix < W) {
            v = x[in_base + iy * W + ix];
        }
        sh[sy * shW + sx] = v;
    }
    __syncthreads();

    const int oy = oy0 + ty;
    const int ox = ox0 + tx;
    if (oy < Hout && ox < Wout) {
        const float* wc = w + c * 9;

        // Shared indices correspond to input at (oy,ox) -> sh[ty+0, tx+0]
        const int s0 = (ty + 0) * shW + (tx + 0);
        // Unrolled 3x3
        float acc = 0.0f;
        acc += sh[s0 + 0] * wc[0];
        acc += sh[s0 + 1] * wc[1];
        acc += sh[s0 + 2] * wc[2];

        acc += sh[s0 + shW + 0] * wc[3];
        acc += sh[s0 + shW + 1] * wc[4];
        acc += sh[s0 + shW + 2] * wc[5];

        acc += sh[s0 + 2*shW + 0] * wc[6];
        acc += sh[s0 + 2*shW + 1] * wc[7];
        acc += sh[s0 + 2*shW + 2] * wc[8];

        const int out_base = ((n * C + c) * Hout) * Wout;
        y[out_base + oy * Wout + ox] = acc;
    }
}

torch::Tensor dwconv3x3_s1p0_fwd(torch::Tensor x, torch::Tensor weight) {
    CHECK_CUDA(x);
    CHECK_CUDA(weight);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(weight);
    CHECK_FLOAT(x);
    CHECK_FLOAT(weight);

    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK(weight.dim() == 4, "weight must be (C,1,3,3)");

    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);

    TORCH_CHECK(weight.size(0) == C && weight.size(1) == 1 && weight.size(2) == 3 && weight.size(3) == 3,
                "Only depthwise 3x3 weights supported");
    TORCH_CHECK(H >= 3 && W >= 3, "Input too small");

    auto y = torch::empty({N, C, H - 2, W - 2}, x.options());

    constexpr int TILE_W = 16;
    constexpr int TILE_H = 16;

    dim3 block(TILE_W, TILE_H, 1);
    dim3 grid((unsigned)((W - 2 + TILE_W - 1) / TILE_W),
              (unsigned)((H - 2 + TILE_H - 1) / TILE_H),
              (unsigned)(N * C));

    hipLaunchKernelGGL((dwconv3x3_s1p0_fwd_kernel<TILE_W, TILE_H>),
                       grid, block, 0, 0,
                       (const float*)x.data_ptr<float>(),
                       (const float*)weight.data_ptr<float>(),
                       (float*)y.data_ptr<float>(),
                       (int)N, (int)C, (int)H, (int)W);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dwconv3x3_s1p0_fwd", &dwconv3x3_s1p0_fwd, "Depthwise conv 3x3 s1 p0 forward (FP32)");
}
"""

_ext = load_inline(
    name="dwconv3x3_s1p0_ext",
    cpp_sources=src,
    functions=None,
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super().__init__()
        # Keep the original module for weight initialization/compat
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding,
                                groups=in_channels, bias=bias)
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fast path for benchmark configuration: k=3, stride=1, padding=0, no bias, groups=in_channels
        if (x.is_cuda and x.dtype == torch.float32 and x.is_contiguous() and
            self.kernel_size == 3 and self.stride == 1 and self.padding == 0 and (not self.bias)):
            w = self.conv2d.weight
            if not w.is_contiguous():
                w = w.contiguous()
            return _ext.dwconv3x3_s1p0_fwd(x, w)
        # Fallback
        return self.conv2d(x)
