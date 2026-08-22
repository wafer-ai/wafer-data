import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Ensure HIP compilation
os.environ.setdefault("CXX", "hipcc")

hip_src = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>

// Simple fused GroupNorm + (groupnorm affine) + external scale (per-channel) for NCHW FP32
// Assumes contiguous NCHW.

__device__ __forceinline__ float warp_reduce_sum(float val) {
    // warpSize is 64 on AMD
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

template<int BLOCK>
__global__ void groupnorm_scale_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ scale,
    float* __restrict__ y,
    int N, int C, int H, int W,
    int G, float eps)
{
    int ng = (int)blockIdx.x;
    int n = ng / G;
    int g = ng - n * G;
    int Cg = C / G;
    int HW = H * W;
    int group_elems = Cg * HW;

    // First pass: compute sum and sumsq
    float sum = 0.0f;
    float sumsq = 0.0f;

    int tid = (int)threadIdx.x;
    for (int i = tid; i < group_elems; i += BLOCK) {
        int c_in_g = i / HW;
        int hw = i - c_in_g * HW;
        int c = g * Cg + c_in_g;
        int idx = ((n * C + c) * H * W) + hw;
        float v = x[idx];
        sum += v;
        sumsq += v * v;
    }

    __shared__ float sh_sum[BLOCK];
    __shared__ float sh_sumsq[BLOCK];
    sh_sum[tid] = sum;
    sh_sumsq[tid] = sumsq;
    __syncthreads();

    // Block reduce
    for (int offset = BLOCK / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sh_sum[tid] += sh_sum[tid + offset];
            sh_sumsq[tid] += sh_sumsq[tid + offset];
        }
        __syncthreads();
    }

    float mean = sh_sum[0] / (float)group_elems;
    float var = sh_sumsq[0] / (float)group_elems - mean * mean;
    float inv_std = rsqrtf(var + eps);

    // Second pass: normalize + affine + scale
    for (int i = tid; i < group_elems; i += BLOCK) {
        int c_in_g = i / HW;
        int hw = i - c_in_g * HW;
        int c = g * Cg + c_in_g;
        int idx = ((n * C + c) * H * W) + hw;
        float v = x[idx];
        float gn = (v - mean) * inv_std;
        float gamma = weight ? weight[c] : 1.0f;
        float beta = bias ? bias[c] : 0.0f;
        float sc = scale ? scale[c] : 1.0f;
        y[idx] = gn * (gamma * sc) + beta;
    }
}

__global__ void clamp_kernel(const float* __restrict__ x, float* __restrict__ y, int size, float lo, float hi) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < size) {
        float v = x[idx];
        v = v < lo ? lo : v;
        v = v > hi ? hi : v;
        y[idx] = v;
    }
}

torch::Tensor groupnorm_scale_hip(torch::Tensor x,
                                 torch::Tensor weight,
                                 torch::Tensor bias,
                                 torch::Tensor scale,
                                 int64_t num_groups,
                                 double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA/HIP tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "FP32 only");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous NCHW");
    auto y = torch::empty_like(x);

    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    int G = (int)num_groups;
    TORCH_CHECK(C % G == 0, "C must be divisible by num_groups");

    const int BLOCK = 256;
    dim3 block(BLOCK);
    dim3 grid((unsigned)(N * G));

    const float* wptr = (weight.defined() && weight.numel() > 0) ? weight.data_ptr<float>() : nullptr;
    const float* bptr = (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr;
    const float* sptr = (scale.defined() && scale.numel() > 0) ? scale.data_ptr<float>() : nullptr;

    hipLaunchKernelGGL((groupnorm_scale_kernel<BLOCK>), grid, block, 0, 0,
        x.data_ptr<float>(), wptr, bptr, sptr, y.data_ptr<float>(),
        N, C, H, W, G, (float)eps);
    return y;
}

torch::Tensor clamp_hip(torch::Tensor x, double lo, double hi) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA/HIP tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "FP32 only");
    auto y = torch::empty_like(x);
    int size = (int)x.numel();
    int block = 256;
    int grid = (size + block - 1) / block;
    hipLaunchKernelGGL(clamp_kernel, dim3(grid), dim3(block), 0, 0,
                       x.data_ptr<float>(), y.data_ptr<float>(), size, (float)lo, (float)hi);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("groupnorm_scale_hip", &groupnorm_scale_hip, "groupnorm+scale (HIP)");
    m.def("clamp_hip", &clamp_hip, "clamp (HIP)");
}
'''

ext = load_inline(
    name='gn_scale_clamp_ext',
    cpp_sources='',
    cuda_sources=hip_src,
    functions=None,
    extra_cuda_cflags=['-O3'],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = self.conv(x)
        # Fused groupnorm + per-channel scale
        x = ext.groupnorm_scale_hip(x.contiguous(), self.group_norm.weight, self.group_norm.bias,
                                   self.scale.view(-1).contiguous(), self.group_norm.num_groups, self.group_norm.eps)
        x = self.maxpool(x)
        x = ext.clamp_hip(x, float(self.clamp_min), float(self.clamp_max))
        return x


def get_inputs():
    batch_size = 128
    in_channels = 8
    height, width = 128, 128
    return [torch.rand(batch_size, in_channels, height, width, device='cuda', dtype=torch.float32)]


def get_init_inputs():
    in_channels = 8
    out_channels = 64
    kernel_size = 3
    num_groups = 16
    scale_shape = (out_channels, 1, 1)
    maxpool_kernel_size = 4
    clamp_min = 0.0
    clamp_max = 1.0
    return [in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max]
