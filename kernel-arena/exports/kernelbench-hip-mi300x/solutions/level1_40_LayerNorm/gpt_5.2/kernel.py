import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Ensure hipcc is used for ROCm builds
os.environ.setdefault("CXX", "hipcc")

# HIP/ROCm implementation of LayerNorm for FP32 inputs.
# Workload: x: [B, 64, 256, 256], normalized over (64,256,256) per batch element.

cpp_src = r'''
#include <torch/extension.h>

torch::Tensor layernorm_forward_hip(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps);
'''

hip_src = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

// Simple block reduction in shared memory
template<int BLOCK>
__device__ __forceinline__ void block_reduce_sum2(float &v0, float &v1) {
    __shared__ float sh0[BLOCK];
    __shared__ float sh1[BLOCK];
    int tid = (int)threadIdx.x;
    sh0[tid] = v0;
    sh1[tid] = v1;
    __syncthreads();
    #pragma unroll
    for (int offset = BLOCK / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sh0[tid] += sh0[tid + offset];
            sh1[tid] += sh1[tid + offset];
        }
        __syncthreads();
    }
    v0 = sh0[0];
    v1 = sh1[0];
}

template<int BLOCK>
__global__ void layernorm_sum_sumsq_f32_vec4(
    const float* __restrict__ x,
    float* __restrict__ sum,
    float* __restrict__ sumsq,
    int64_t N_vec4, // N/4
    int64_t N       // N
) {
    // grid: (B, blocks_per_batch)
    int b = (int)blockIdx.x;
    int bid = (int)blockIdx.y;
    int tid = (int)threadIdx.x;

    const float4* x4 = reinterpret_cast<const float4*>(x + (int64_t)b * N);

    int64_t idx = (int64_t)(bid * BLOCK + tid);
    int64_t stride = (int64_t)gridDim.y * BLOCK;

    float s = 0.0f;
    float ss = 0.0f;

    for (int64_t i = idx; i < N_vec4; i += stride) {
        float4 v = x4[i];
        s  += v.x + v.y + v.z + v.w;
        ss += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }

    block_reduce_sum2<BLOCK>(s, ss);
    if (tid == 0) {
        atomicAdd(sum + b, s);
        atomicAdd(sumsq + b, ss);
    }
}

__global__ void layernorm_finalize_stats_f32(
    const float* __restrict__ sum,
    const float* __restrict__ sumsq,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    int B,
    float invN,
    float eps
) {
    int b = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (b < B) {
        float m = sum[b] * invN;
        float ex2 = sumsq[b] * invN;
        float var = ex2 - m * m;
        var = var < 0.0f ? 0.0f : var;
        mean[b] = m;
        rstd[b] = rsqrtf(var + eps);
    }
}

template<int BLOCK>
__global__ void layernorm_apply_f32_vec4(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ out,
    int64_t N_vec4,
    int64_t N
) {
    // grid: (blocks_x, B)
    int b = (int)blockIdx.y;
    int tid = (int)threadIdx.x;

    float m = mean[b];
    float rs = rstd[b];

    const float4* x4 = reinterpret_cast<const float4*>(x + (int64_t)b * N);
    const float4* w4 = reinterpret_cast<const float4*>(weight);
    const float4* b4 = reinterpret_cast<const float4*>(bias);
    float4* o4 = reinterpret_cast<float4*>(out + (int64_t)b * N);

    int64_t idx = (int64_t)blockIdx.x * BLOCK + tid;
    int64_t stride = (int64_t)gridDim.x * BLOCK;

    for (int64_t i = idx; i < N_vec4; i += stride) {
        float4 xv = x4[i];
        float4 wv = w4[i];
        float4 bv = b4[i];

        float4 y;
        y.x = (xv.x - m) * rs * wv.x + bv.x;
        y.y = (xv.y - m) * rs * wv.y + bv.y;
        y.z = (xv.z - m) * rs * wv.z + bv.z;
        y.w = (xv.w - m) * rs * wv.w + bv.w;

        o4[i] = y;
    }
}

torch::Tensor layernorm_forward_hip(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA/HIP tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.is_cuda() && bias.is_cuda(), "weight/bias must be CUDA/HIP tensors");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32 && bias.scalar_type() == torch::kFloat32,
                "weight/bias must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous() && bias.is_contiguous(), "weight/bias must be contiguous");

    TORCH_CHECK(x.dim() == 4, "expected 4D input");
    int B = (int)x.size(0);
    int64_t N = x.size(1) * x.size(2) * x.size(3);
    TORCH_CHECK(weight.numel() == N && bias.numel() == N, "weight/bias numel mismatch");
    TORCH_CHECK((N % 4) == 0, "N must be divisible by 4 for vec4 path");

    auto out = torch::empty_like(x);
    auto sum = torch::zeros({B}, x.options());
    auto sumsq = torch::zeros({B}, x.options());
    auto mean = torch::empty({B}, x.options());
    auto rstd = torch::empty({B}, x.options());

    constexpr int BLOCK = 256;
    int blocks_per_batch = 256;

    cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

    int64_t N_vec4 = N / 4;

    dim3 grid1((unsigned)B, (unsigned)blocks_per_batch, 1);
    dim3 block1(BLOCK, 1, 1);
    hipLaunchKernelGGL((layernorm_sum_sumsq_f32_vec4<BLOCK>), grid1, block1, 0, stream,
                       (const float*)x.data_ptr<float>(),
                       (float*)sum.data_ptr<float>(),
                       (float*)sumsq.data_ptr<float>(),
                       N_vec4,
                       N);

    // Finalize stats
    dim3 grid2((unsigned)((B + 255) / 256), 1, 1);
    dim3 block2(256, 1, 1);
    float invN = 1.0f / (float)N;
    hipLaunchKernelGGL(layernorm_finalize_stats_f32, grid2, block2, 0, stream,
                       (const float*)sum.data_ptr<float>(),
                       (const float*)sumsq.data_ptr<float>(),
                       (float*)mean.data_ptr<float>(),
                       (float*)rstd.data_ptr<float>(),
                       B,
                       invN,
                       (float)eps);

    // Apply
    int blocks_x = (int)((N_vec4 + BLOCK - 1) / BLOCK);
    if (blocks_x > 65535) blocks_x = 65535;
    dim3 grid3((unsigned)blocks_x, (unsigned)B, 1);
    dim3 block3(BLOCK, 1, 1);

    hipLaunchKernelGGL((layernorm_apply_f32_vec4<BLOCK>), grid3, block3, 0, stream,
                       (const float*)x.data_ptr<float>(),
                       (const float*)weight.data_ptr<float>(),
                       (const float*)bias.data_ptr<float>(),
                       (const float*)mean.data_ptr<float>(),
                       (const float*)rstd.data_ptr<float>(),
                       (float*)out.data_ptr<float>(),
                       N_vec4,
                       N);

    return out;
}
'''

_layernorm_ext = load_inline(
    name="layernorm_f32_rocm_ext",
    cpp_sources=cpp_src,
    cuda_sources=hip_src,
    functions=["layernorm_forward_hip"],
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """Optimized LayerNorm using custom HIP/ROCm kernels (FP32)."""

    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)
        self.eps = 1e-5
        # Match nn.LayerNorm parameterization
        self.weight = nn.Parameter(torch.ones(self.normalized_shape, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and x.dtype == torch.float32:
            return _layernorm_ext.layernorm_forward_hip(x, self.weight, self.bias, self.eps)
        return torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
