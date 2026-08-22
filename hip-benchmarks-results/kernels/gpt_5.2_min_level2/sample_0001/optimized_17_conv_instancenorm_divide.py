import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ.setdefault("CXX", "hipcc")

instancenorm_div_cpp = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>

struct WelfordData {
    float mean;
    float m2;
    int count;
};

__device__ __forceinline__ WelfordData welford_combine(const WelfordData &a, const WelfordData &b) {
    if (a.count == 0) return b;
    if (b.count == 0) return a;
    WelfordData out;
    out.count = a.count + b.count;
    float delta = b.mean - a.mean;
    out.mean = a.mean + delta * (float)b.count / (float)out.count;
    out.m2 = a.m2 + b.m2 + delta * delta * (float)a.count * (float)b.count / (float)out.count;
    return out;
}

__global__ void instancenorm_div_kernel(const float* __restrict__ x,
                                       float* __restrict__ y,
                                       int N, int C, int H, int W,
                                       float eps,
                                       float inv_divide) {
    int nc = (int)blockIdx.x;
    int n = nc / C;
    int c = nc - n * C;
    int HW = H * W;
    int tid = (int)threadIdx.x;

    int base = ((n * C + c) * H) * W;

    WelfordData wd;
    wd.mean = 0.0f;
    wd.m2 = 0.0f;
    wd.count = 0;

    for (int idx = tid; idx < HW; idx += (int)blockDim.x) {
        float v = x[base + idx];
        wd.count += 1;
        float delta = v - wd.mean;
        wd.mean += delta / (float)wd.count;
        float delta2 = v - wd.mean;
        wd.m2 += delta * delta2;
    }

    extern __shared__ unsigned char smem[];
    WelfordData* sdata = reinterpret_cast<WelfordData*>(smem);
    sdata[tid] = wd;
    __syncthreads();

    for (int offset = (int)blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] = welford_combine(sdata[tid], sdata[tid + offset]);
        }
        __syncthreads();
    }

    float mean = sdata[0].mean;
    float var = sdata[0].m2 / (float)HW;
    float inv_std = rsqrtf(var + eps);

    for (int idx = tid; idx < HW; idx += (int)blockDim.x) {
        float v = x[base + idx];
        y[base + idx] = (v - mean) * inv_std * inv_divide;
    }
}

torch::Tensor instancenorm_div_hip(torch::Tensor x, double eps, double inv_divide) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA/HIP tensor");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int N = (int)x.size(0);
    int C = (int)x.size(1);
    int H = (int)x.size(2);
    int W = (int)x.size(3);

    auto y = torch::empty_like(x);

    int blocks = N * C;
    int threads = 256;
    size_t shmem = (size_t)threads * sizeof(WelfordData);

    hipStream_t stream = at::hip::getDefaultHIPStream();
    hipLaunchKernelGGL(instancenorm_div_kernel,
                       dim3(blocks), dim3(threads), shmem, stream,
                       (const float*)x.data_ptr<float>(),
                       (float*)y.data_ptr<float>(),
                       N, C, H, W,
                       (float)eps,
                       (float)inv_divide);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("instancenorm_div_hip", &instancenorm_div_hip, "Fused InstanceNorm+Divide (HIP)");
}
"""

instancenorm_div_ext = load_inline(
    name="instancenorm_div_ext",
    cpp_sources=instancenorm_div_cpp,
    functions=None,
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.eps = 1e-5
        self.inv_divide = 1.0 / float(divide_by)

    def forward(self, x):
        x = self.conv(x)
        if not x.is_contiguous():
            x = x.contiguous()
        return instancenorm_div_ext.instancenorm_div_hip(x, self.eps, self.inv_divide)


batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
divide_by = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divide_by]
