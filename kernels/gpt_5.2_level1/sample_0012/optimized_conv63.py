import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ.setdefault("CXX", "hipcc")

cpp_src = r'''
#include <torch/extension.h>

torch::Tensor conv3x3_forward(torch::Tensor input, torch::Tensor weight);
'''

cuda_src = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA/HIP tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")

// Specialized kernel for benchmark shape:
// - NCHW
// - stride=1, padding=0, dilation=1, groups=1
// - IC=16, KH=KW=3
// Strategy:
// - Each thread computes 2 output channels (OC_BLK=2) for one spatial output (y,x)
//   to reuse the same input patch.
// - Tile: 16x8 spatial, 128 threads per block (2 wavefronts).

constexpr int IC = 16;
constexpr int KH = 3;
constexpr int KW = 3;
constexpr int TILE_H = 8;
constexpr int TILE_W = 16;
constexpr int OC_BLK = 2;
constexpr int IN_TILE_H = TILE_H + KH - 1; // 10
constexpr int IN_TILE_W = TILE_W + KW - 1; // 18

__global__ void conv3x3_nchw_ic16_oc2_tile16x8_2oc_per_thread(
    const float* __restrict__ inp,
    const float* __restrict__ w,
    float* __restrict__ out,
    int H,
    int W,
    int OC)
{
    const int Wout = W - 2;
    const int Hout = H - 2;

    const int oc_blocks = (OC + OC_BLK - 1) / OC_BLK;
    const int n = (int)(blockIdx.z / oc_blocks);
    const int oc_blk = (int)(blockIdx.z - (unsigned)(n * oc_blocks));
    const int oc_base = oc_blk * OC_BLK;

    const int out_x = (int)blockIdx.x * TILE_W + (int)threadIdx.x;
    const int out_y = (int)blockIdx.y * TILE_H + (int)threadIdx.y;

    __shared__ float sh_in[IC * IN_TILE_H * IN_TILE_W];     // 16*180
    __shared__ float sh_w[OC_BLK * IC * KH * KW];          // 2*16*9

    const int tid = (int)threadIdx.y * TILE_W + (int)threadIdx.x; // 0..127

    // Load weights (288 floats)
    for (int idx = tid; idx < OC_BLK * IC * KH * KW; idx += TILE_W * TILE_H) {
        const int oc_local = idx / (IC * KH * KW);
        const int rem0 = idx - oc_local * (IC * KH * KW);
        const int ic = rem0 / (KH * KW);
        const int kk = rem0 - ic * (KH * KW);
        const int oc_g = oc_base + oc_local;
        float val = 0.0f;
        if (oc_g < OC) {
            val = w[((oc_g * IC + ic) * KH * KW) + kk];
        }
        sh_w[idx] = val;
    }

    // Load input tile for each ic (180 floats/channel)
    const int in_tile_elems = IN_TILE_H * IN_TILE_W; // 180
    const int tile_in_x0 = (int)blockIdx.x * TILE_W;
    const int tile_in_y0 = (int)blockIdx.y * TILE_H;

    #pragma unroll
    for (int ic = 0; ic < IC; ++ic) {
        for (int idx = tid; idx < in_tile_elems; idx += TILE_W * TILE_H) {
            const int iy = idx / IN_TILE_W;
            const int ix = idx - iy * IN_TILE_W;
            const int in_y = tile_in_y0 + iy;
            const int in_x = tile_in_x0 + ix;
            float v = 0.0f;
            if ((unsigned)in_y < (unsigned)H && (unsigned)in_x < (unsigned)W) {
                v = inp[(((n * IC + ic) * H + in_y) * W) + in_x];
            }
            sh_in[ic * in_tile_elems + idx] = v;
        }
    }

    __syncthreads();

    if ((unsigned)out_y >= (unsigned)Hout || (unsigned)out_x >= (unsigned)Wout) return;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    const int base_in = (int)threadIdx.y * IN_TILE_W + (int)threadIdx.x;

    #pragma unroll
    for (int ic = 0; ic < IC; ++ic) {
        const float* in_ptr = sh_in + ic * in_tile_elems + base_in;

        const float i00 = in_ptr[0];
        const float i01 = in_ptr[1];
        const float i02 = in_ptr[2];
        const float i10 = in_ptr[IN_TILE_W + 0];
        const float i11 = in_ptr[IN_TILE_W + 1];
        const float i12 = in_ptr[IN_TILE_W + 2];
        const float i20 = in_ptr[2 * IN_TILE_W + 0];
        const float i21 = in_ptr[2 * IN_TILE_W + 1];
        const float i22 = in_ptr[2 * IN_TILE_W + 2];

        const float* w0 = sh_w + (0 * IC + ic) * 9;
        const float* w1 = sh_w + (1 * IC + ic) * 9;

        // oc0
        acc0 = fmaf(i00, w0[0], acc0);
        acc0 = fmaf(i01, w0[1], acc0);
        acc0 = fmaf(i02, w0[2], acc0);
        acc0 = fmaf(i10, w0[3], acc0);
        acc0 = fmaf(i11, w0[4], acc0);
        acc0 = fmaf(i12, w0[5], acc0);
        acc0 = fmaf(i20, w0[6], acc0);
        acc0 = fmaf(i21, w0[7], acc0);
        acc0 = fmaf(i22, w0[8], acc0);

        // oc1
        acc1 = fmaf(i00, w1[0], acc1);
        acc1 = fmaf(i01, w1[1], acc1);
        acc1 = fmaf(i02, w1[2], acc1);
        acc1 = fmaf(i10, w1[3], acc1);
        acc1 = fmaf(i11, w1[4], acc1);
        acc1 = fmaf(i12, w1[5], acc1);
        acc1 = fmaf(i20, w1[6], acc1);
        acc1 = fmaf(i21, w1[7], acc1);
        acc1 = fmaf(i22, w1[8], acc1);
    }

    const int plane = Hout * Wout;
    const int base = (((n * OC) * Hout + out_y) * Wout) + out_x;

    const int oc0 = oc_base + 0;
    const int oc1 = oc_base + 1;
    if (oc0 < OC) out[base + oc0 * plane] = acc0;
    if (oc1 < OC) out[base + oc1 * plane] = acc1;
}

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

torch::Tensor conv3x3_forward(torch::Tensor input, torch::Tensor weight) {
    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(weight);
    CHECK_FLOAT(input);
    CHECK_FLOAT(weight);

    TORCH_CHECK(input.dim() == 4, "input must be NCHW");
    TORCH_CHECK(weight.dim() == 4, "weight must be OIHW");
    TORCH_CHECK(input.size(1) == IC, "Only in_channels=16 supported");
    TORCH_CHECK(weight.size(1) == IC && weight.size(2) == KH && weight.size(3) == KW,
                "Only 3x3 kernels with IC=16 supported");

    const int64_t N = input.size(0);
    const int64_t H = input.size(2);
    const int64_t W = input.size(3);
    const int64_t OC = weight.size(0);

    TORCH_CHECK(H >= 3 && W >= 3, "H/W must be >= 3");

    auto out = torch::empty({N, OC, H - 2, W - 2}, input.options());

    const int Hout = (int)(H - 2);
    const int Wout = (int)(W - 2);

    dim3 block(TILE_W, TILE_H, 1); // 16x8
    dim3 grid(ceil_div(Wout, TILE_W), ceil_div(Hout, TILE_H), (unsigned)(N * ((OC + OC_BLK - 1) / OC_BLK)));

    hipStream_t stream = (hipStream_t)at::cuda::getDefaultCUDAStream().stream();

    hipLaunchKernelGGL(
        conv3x3_nchw_ic16_oc2_tile16x8_2oc_per_thread,
        grid,
        block,
        0,
        stream,
        (const float*)input.data_ptr<float>(),
        (const float*)weight.data_ptr<float>(),
        (float*)out.data_ptr<float>(),
        (int)H,
        (int)W,
        (int)OC);

    return out;
}
'''

conv3x3_ext = load_inline(
    name="conv3x3_ic16_k3_s1_p0_oc2_tile16x8_v3",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["conv3x3_forward"],
    with_cuda=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        # Keep reference parameter initialization identical
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.is_cuda
            and x.dtype == torch.float32
            and x.is_contiguous()
            and self.conv2d.weight.is_contiguous()
            and x.dim() == 4
            and x.size(1) == 16
            and self.conv2d.weight.size(2) == 3
            and self.conv2d.weight.size(3) == 3
            and self.conv2d.stride == (1, 1)
            and self.conv2d.padding == (0, 0)
            and self.conv2d.dilation == (1, 1)
            and self.conv2d.groups == 1
            and self.conv2d.bias is None
        ):
            return conv3x3_ext.conv3x3_forward(x, self.conv2d.weight)

        return self.conv2d(x)
