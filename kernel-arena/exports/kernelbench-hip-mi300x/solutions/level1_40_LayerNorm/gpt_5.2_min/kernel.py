import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ.setdefault("CXX", "hipcc")

layernorm_cpp = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>

template <int TPB>
__device__ __forceinline__ void block_reduce_sum2(float &sum, float &sumsq) {
    __shared__ float sh_sum[TPB];
    __shared__ float sh_sumsq[TPB];
    int tid = threadIdx.x;
    sh_sum[tid] = sum;
    sh_sumsq[tid] = sumsq;
    __syncthreads();

    for (int offset = TPB / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sh_sum[tid] += sh_sum[tid + offset];
            sh_sumsq[tid] += sh_sumsq[tid + offset];
        }
        __syncthreads();
    }

    sum = sh_sum[0];
    sumsq = sh_sumsq[0];
}

template <int TPB>
__global__ void partial_sum_kernel(const float* __restrict__ x,
                                   float* __restrict__ partial_sum,
                                   float* __restrict__ partial_sumsq,
                                   int N, int num_blocks) {
    int blk = (int)blockIdx.x;
    int b = (int)blockIdx.y;
    int tid = (int)threadIdx.x;

    int64_t base = (int64_t)b * (int64_t)N;

    int64_t start = (int64_t)blk * (int64_t)TPB;
    int64_t idx = start + tid;

    float sum = 0.0f;
    float sumsq = 0.0f;

    int64_t stride = (int64_t)TPB * (int64_t)num_blocks;
    for (int64_t i = idx; i < N; i += stride) {
        float v = x[base + i];
        sum += v;
        sumsq += v * v;
    }

    block_reduce_sum2<TPB>(sum, sumsq);
    if (tid == 0) {
        partial_sum[b * num_blocks + blk] = sum;
        partial_sumsq[b * num_blocks + blk] = sumsq;
    }
}

template <int TPB>
__global__ void final_stats_kernel(const float* __restrict__ partial_sum,
                                   const float* __restrict__ partial_sumsq,
                                   float* __restrict__ mean,
                                   float* __restrict__ invstd,
                                   int N, int num_blocks, float eps) {
    int b = (int)blockIdx.x;
    int tid = (int)threadIdx.x;

    float sum = 0.0f;
    float sumsq = 0.0f;

    for (int i = tid; i < num_blocks; i += TPB) {
        sum += partial_sum[b * num_blocks + i];
        sumsq += partial_sumsq[b * num_blocks + i];
    }

    block_reduce_sum2<TPB>(sum, sumsq);

    if (tid == 0) {
        float m = sum / (float)N;
        float var = sumsq / (float)N - m * m;
        var = var < 0.0f ? 0.0f : var;
        mean[b] = m;
        invstd[b] = rsqrtf(var + eps);
    }
}

template <int TPB>
__global__ void layernorm_affine_kernel(const float* __restrict__ x,
                                       const float* __restrict__ weight,
                                       const float* __restrict__ bias,
                                       const float* __restrict__ mean,
                                       const float* __restrict__ invstd,
                                       float* __restrict__ y,
                                       int N) {
    int idx = (int)blockIdx.x * TPB + (int)threadIdx.x;
    int b = (int)blockIdx.y;
    if (idx >= N) return;

    int64_t base = (int64_t)b * (int64_t)N;
    float v = x[base + idx];
    float n = (v - mean[b]) * invstd[b];
    y[base + idx] = n * weight[idx] + bias[idx];
}

torch::Tensor layernorm_forward_hip(torch::Tensor x,
                                   torch::Tensor weight,
                                   torch::Tensor bias,
                                   double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA/HIP tensor");
    TORCH_CHECK(weight.is_cuda() && bias.is_cuda(), "weight/bias must be CUDA/HIP tensors");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat && bias.scalar_type() == at::kFloat, "weight/bias must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous() && bias.is_contiguous(), "weight/bias must be contiguous");

    int B = (int)x.size(0);
    int64_t N64 = 1;
    for (int i = 1; i < x.dim(); i++) N64 *= x.size(i);
    TORCH_CHECK(N64 <= INT_MAX, "N too large");
    int N = (int)N64;

    auto y = torch::empty_like(x);

    constexpr int TPB = 256;
    int num_blocks = (N + TPB - 1) / TPB;
    if (num_blocks > 1024) num_blocks = 1024;
    if (num_blocks < 1) num_blocks = 1;

    auto opts = torch::TensorOptions().device(x.device()).dtype(torch::kFloat);
    auto partial_sum = torch::empty({B, num_blocks}, opts);
    auto partial_sumsq = torch::empty({B, num_blocks}, opts);
    auto mean = torch::empty({B}, opts);
    auto invstd = torch::empty({B}, opts);

    hipStream_t stream = at::hip::getDefaultHIPStream();

    dim3 grid1(num_blocks, B, 1);
    dim3 block(TPB, 1, 1);
    hipLaunchKernelGGL((partial_sum_kernel<TPB>), grid1, block, 0, stream,
                      (const float*)x.data_ptr<float>(),
                      (float*)partial_sum.data_ptr<float>(),
                      (float*)partial_sumsq.data_ptr<float>(),
                      N, num_blocks);

    dim3 grid2(B, 1, 1);
    hipLaunchKernelGGL((final_stats_kernel<TPB>), grid2, block, 0, stream,
                      (const float*)partial_sum.data_ptr<float>(),
                      (const float*)partial_sumsq.data_ptr<float>(),
                      (float*)mean.data_ptr<float>(),
                      (float*)invstd.data_ptr<float>(),
                      N, num_blocks, (float)eps);

    dim3 grid3((N + TPB - 1) / TPB, B, 1);
    hipLaunchKernelGGL((layernorm_affine_kernel<TPB>), grid3, block, 0, stream,
                      (const float*)x.data_ptr<float>(),
                      (const float*)weight.data_ptr<float>(),
                      (const float*)bias.data_ptr<float>(),
                      (const float*)mean.data_ptr<float>(),
                      (const float*)invstd.data_ptr<float>(),
                      (float*)y.data_ptr<float>(),
                      N);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm_forward_hip", &layernorm_forward_hip, "LayerNorm forward (HIP)");
}
"""

layernorm_ext = load_inline(
    name="layernorm_ext_rocm",
    cpp_sources=layernorm_cpp,
    functions=None,
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)
        N = 1
        for d in self.normalized_shape:
            N *= d
        self.weight = nn.Parameter(torch.ones(N, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(N, dtype=torch.float32))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            return torch.nn.functional.layer_norm(
                x,
                self.normalized_shape,
                self.weight.view(self.normalized_shape),
                self.bias.view(self.normalized_shape),
                self.eps,
            )
        return layernorm_ext.layernorm_forward_hip(
            x.contiguous(), self.weight.contiguous(), self.bias.contiguous(), self.eps
        )
