import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Force HIP compilation for ROCm
os.environ.setdefault("CXX", "hipcc")

# Specialized maxpool2d for: kernel=4, stride=1, padding=1, dilation=1, FP32, NCHW contiguous
# Uses shared memory tiling to reuse input loads across overlapping windows.

hip_source = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cfloat>

// Tile sizes (output tile): TH x TW
// Shared tile size: (TH + K - 1) x (TW + K - 1) because stride=1, dilation=1
static constexpr int K = 4;
static constexpr int PAD = 1;
static constexpr int STRIDE = 1;
static constexpr int DIL = 1;

static constexpr int TH = 8;
static constexpr int TW = 32;
static constexpr int SH = TH + (K - 1); // 11
static constexpr int SW = TW + (K - 1); // 35

__global__ void maxpool2d_4x4s1p1_fp32_nchw_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int H, int W,
    int Hout, int Wout,
    int C
) {
    // grid.z indexes combined (n,c)
    int nc = (int)blockIdx.z;
    int n = nc / C;
    int c = nc - n * C;

    int ox = (int)blockIdx.x * TW + (int)threadIdx.x;
    int oy = (int)blockIdx.y * TH + (int)threadIdx.y;

    // Shared tile base input coordinate
    int base_ix = (int)blockIdx.x * TW - PAD;
    int base_iy = (int)blockIdx.y * TH - PAD;

    extern __shared__ float sh[]; // size SH*SW

    // Cooperative load of SH*SW values
    int tid = (int)threadIdx.y * (int)blockDim.x + (int)threadIdx.x;
    int nthreads = (int)blockDim.x * (int)blockDim.y;

    // Base pointer for (n,c) plane
    // Index = (((n*C + c)*H) + iy)*W + ix
    long long plane_offset = ((long long)n * (long long)C + (long long)c) * (long long)H * (long long)W;

    for (int i = tid; i < SH * SW; i += nthreads) {
        int ty = i / SW;
        int tx = i - ty * SW;
        int iy = base_iy + ty;
        int ix = base_ix + tx;
        float v = -FLT_MAX;
        if ((unsigned)iy < (unsigned)H && (unsigned)ix < (unsigned)W) {
            v = x[plane_offset + (long long)iy * (long long)W + (long long)ix];
        }
        sh[i] = v;
    }

    __syncthreads();

    if (ox >= Wout || oy >= Hout) return;

    int ly = (int)threadIdx.y;
    int lx = (int)threadIdx.x;

    // Unrolled 4x4 max over shared tile
    float m = -FLT_MAX;
#pragma unroll
    for (int ky = 0; ky < K; ky++) {
        int row = (ly + ky) * SW + lx;
        float v0 = sh[row + 0];
        float v1 = sh[row + 1];
        float v2 = sh[row + 2];
        float v3 = sh[row + 3];
        float r = fmaxf(fmaxf(v0, v1), fmaxf(v2, v3));
        m = fmaxf(m, r);
    }

    long long out_offset = ((long long)n * (long long)C + (long long)c) * (long long)Hout * (long long)Wout;
    y[out_offset + (long long)oy * (long long)Wout + (long long)ox] = m;
}

torch::Tensor maxpool2d_4x4s1p1_fp32(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA/ROCm tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous NCHW");

    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);

    // For K=4, stride=1, pad=1, dilation=1
    const int64_t Hout = (H + 2 * PAD - DIL * (K - 1) - 1) / STRIDE + 1;
    const int64_t Wout = (W + 2 * PAD - DIL * (K - 1) - 1) / STRIDE + 1;

    auto y = torch::empty({N, C, Hout, Wout}, x.options());

    c10::cuda::CUDAGuard device_guard(x.device());
    hipStream_t stream = (hipStream_t)at::cuda::getDefaultCUDAStream();

    dim3 block(TW, TH, 1); // 32x8 = 256 threads
    dim3 grid((unsigned)((Wout + TW - 1) / TW), (unsigned)((Hout + TH - 1) / TH), (unsigned)(N * C));
    size_t shmem = (size_t)(SH * SW) * sizeof(float);

    maxpool2d_4x4s1p1_fp32_nchw_kernel<<<grid, block, shmem, stream>>>(
        (const float*)x.data_ptr<float>(),
        (float*)y.data_ptr<float>(),
        (int)H, (int)W,
        (int)Hout, (int)Wout,
        (int)C
    );

    return y;
}
'''

maxpool_ext = load_inline(
    name="maxpool2d_4x4s1p1_fp32_ext",
    cpp_sources=hip_source,
    functions=["maxpool2d_4x4s1p1_fp32"],
    with_cuda=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super().__init__()
        # Fallback for CPU / non-matching configs
        self.maxpool_fallback = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.is_cuda
            and x.dtype == torch.float32
            and x.is_contiguous()
            and self.kernel_size == 4
            and self.stride == 1
            and self.padding == 1
            and self.dilation == 1
            and x.dim() == 4
        ):
            return maxpool_ext.maxpool2d_4x4s1p1_fp32(x)
        return self.maxpool_fallback(x)


# Keep the same input generators
batch_size = 32
channels = 64
height = 512
width = 512
kernel_size = 4
stride = 1
padding = 1
dilation = 1

def get_inputs():
    x = torch.rand(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]
