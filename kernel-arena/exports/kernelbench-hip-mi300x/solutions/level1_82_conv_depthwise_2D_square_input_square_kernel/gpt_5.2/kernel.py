import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Ensure HIP compiler is used on ROCm systems
os.environ.setdefault("CXX", "hipcc")

# Depthwise conv2d, specialized fast path for FP32, stride=1
# Supports padding=0 or padding=1; supports kernel_size=3.

cuda_src = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA/HIP tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")

// Tile sizes
constexpr int TX = 16;
constexpr int TY = 16;

__global__ void dwconv3x3_s1_pN_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b, // can be nullptr
    float* __restrict__ y,
    int N, int C, int H, int W,
    int Hout, int Wout,
    int pad)
{
    // blockIdx.z maps to (n, c)
    int nc = (int)blockIdx.z;
    int c = nc % C;
    int n = nc / C;
    if (n >= N) return;

    int oh = (int)blockIdx.y * TY + (int)threadIdx.y;
    int ow = (int)blockIdx.x * TX + (int)threadIdx.x;

    constexpr int K = 3;
    constexpr int TILE_W = TX + (K - 1);
    constexpr int TILE_H = TY + (K - 1);

    __shared__ float tile[TILE_H * TILE_W];
    __shared__ float wsh[9];

    int tid = (int)threadIdx.y * TX + (int)threadIdx.x;
    if (tid < 9) {
        wsh[tid] = w[c * 9 + tid];
    }

    // Load input tile with padding handled via bounds checks.
    // Input origin for this output tile:
    // in_h0 = oh0 - pad, in_w0 = ow0 - pad
    int oh0 = (int)blockIdx.y * TY;
    int ow0 = (int)blockIdx.x * TX;
    int in_h0 = oh0 - pad;
    int in_w0 = ow0 - pad;

    for (int i = tid; i < TILE_H * TILE_W; i += TX * TY) {
        int th = i / TILE_W;
        int tw = i - th * TILE_W;
        int ih = in_h0 + th;
        int iw = in_w0 + tw;

        float val = 0.0f;
        if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W) {
            // x is NCHW contiguous
            long idx = (((long)n * C + c) * H + ih) * W + iw;
            val = x[idx];
        }
        tile[i] = val;
    }
    __syncthreads();

    if (oh < Hout && ow < Wout) {
        int base = (int)threadIdx.y * TILE_W + (int)threadIdx.x;

        // Unrolled 3x3
        float s0 = tile[base];
        float s1 = tile[base + 1];
        float s2 = tile[base + 2];
        float s3 = tile[base + TILE_W];
        float s4 = tile[base + TILE_W + 1];
        float s5 = tile[base + TILE_W + 2];
        float s6 = tile[base + 2 * TILE_W];
        float s7 = tile[base + 2 * TILE_W + 1];
        float s8 = tile[base + 2 * TILE_W + 2];

        float acc = 0.0f;
        acc = fmaf(wsh[0], s0, acc);
        acc = fmaf(wsh[1], s1, acc);
        acc = fmaf(wsh[2], s2, acc);
        acc = fmaf(wsh[3], s3, acc);
        acc = fmaf(wsh[4], s4, acc);
        acc = fmaf(wsh[5], s5, acc);
        acc = fmaf(wsh[6], s6, acc);
        acc = fmaf(wsh[7], s7, acc);
        acc = fmaf(wsh[8], s8, acc);

        if (b != nullptr) acc += b[c];

        long oidx = (((long)n * C + c) * Hout + oh) * Wout + ow;
        y[oidx] = acc;
    }
}

torch::Tensor dwconv3x3_s1_hip(torch::Tensor x, torch::Tensor weight, int64_t padding, torch::optional<torch::Tensor> bias_opt) {
    CHECK_CUDA(x);
    CHECK_CUDA(weight);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(weight);
    CHECK_FLOAT(x);
    CHECK_FLOAT(weight);

    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK(weight.dim() == 4, "weight must be [C,1,3,3]");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    TORCH_CHECK(weight.size(0) == C && weight.size(1) == 1 && weight.size(2) == 3 && weight.size(3) == 3,
                "weight must be [C,1,3,3]");

    TORCH_CHECK(padding == 0 || padding == 1, "only padding 0 or 1 supported in fast path");

    // stride is fixed at 1 in this specialized kernel
    int64_t Hout = H + 2 * padding - 3 + 1;
    int64_t Wout = W + 2 * padding - 3 + 1;
    TORCH_CHECK(Hout > 0 && Wout > 0, "invalid output size");

    auto y = torch::empty({N, C, Hout, Wout}, x.options());

    const float* bptr = nullptr;
    torch::Tensor bias;
    if (bias_opt.has_value()) {
        bias = bias_opt.value();
        CHECK_CUDA(bias);
        CHECK_CONTIGUOUS(bias);
        CHECK_FLOAT(bias);
        TORCH_CHECK(bias.numel() == C, "bias must have C elements");
        bptr = (const float*)bias.data_ptr<float>();
    }

    dim3 block(TX, TY, 1);
    dim3 grid((unsigned)((Wout + TX - 1) / TX), (unsigned)((Hout + TY - 1) / TY), (unsigned)(N * C));

    auto stream = at::cuda::getDefaultCUDAStream();
    hipStream_t hip_stream = (hipStream_t)stream.stream();

    hipLaunchKernelGGL(dwconv3x3_s1_pN_kernel, grid, block, 0, hip_stream,
                      (const float*)x.data_ptr<float>(),
                      (const float*)weight.data_ptr<float>(),
                      bptr,
                      (float*)y.data_ptr<float>(),
                      (int)N, (int)C, (int)H, (int)W,
                      (int)Hout, (int)Wout,
                      (int)padding);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dwconv3x3_s1_hip", &dwconv3x3_s1_hip, "Depthwise conv2d 3x3 stride1 (HIP)");
}
"""

# Build once (cached by PyTorch)
dwconv_ext = load_inline(
    name="dwconv3x3_s1_hip_ext",
    cpp_sources="",
    cuda_sources=cuda_src,
    functions=None,
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        # Keep the same module/parameter structure for state_dict compatibility
        self.conv2d = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fast path only for the benchmark config (FP32, depthwise, k=3, stride=1)
        if (
            x.is_cuda
            and x.dtype == torch.float32
            and x.is_contiguous()
            and self.conv2d.weight.is_cuda
            and self.conv2d.weight.dtype == torch.float32
            and self.conv2d.weight.is_contiguous()
            and self.conv2d.groups == self.conv2d.in_channels
            and self.conv2d.in_channels == self.conv2d.out_channels
            and self.conv2d.kernel_size == (3, 3)
            and self.conv2d.stride == (1, 1)
            and self.conv2d.dilation == (1, 1)
        ):
            bias = self.conv2d.bias
            return dwconv_ext.dwconv3x3_s1_hip(x, self.conv2d.weight, int(self.conv2d.padding[0]), bias)

        # Fallback to PyTorch for any unsupported case
        return self.conv2d(x)
