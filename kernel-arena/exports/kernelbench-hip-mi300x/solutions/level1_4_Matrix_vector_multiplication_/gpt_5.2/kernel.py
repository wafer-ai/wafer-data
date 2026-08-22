import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Force HIP compilation on ROCm
os.environ.setdefault("CXX", "hipcc")

_cpp_src = r'''
#include <torch/extension.h>

torch::Tensor matvec_fp32_cuda(torch::Tensor A, torch::Tensor B);

torch::Tensor matvec_fp32(torch::Tensor A, torch::Tensor B) {
    return matvec_fp32_cuda(A, B);
}
'''

_cuda_src = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ float wave_reduce_sum(float v) {
    // warpSize is 64 on AMD
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset, warpSize);
    }
    return v;
}

extern "C" __global__ __launch_bounds__(256)
void gemv_fp32_vec4_kernel(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int K4) {
    int row = (int)blockIdx.x;
    int tid = (int)threadIdx.x;

    const float4* __restrict__ A4 = reinterpret_cast<const float4*>(A + ((size_t)row) * ((size_t)K4) * 4);
    const float4* __restrict__ B4 = reinterpret_cast<const float4*>(B);

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

    // Unroll by 4 in the K4 dimension to reduce loop overhead and improve ILP.
    const int stride = (int)blockDim.x * 4;
    int i = tid;
    for (; i + 3 * (int)blockDim.x < K4; i += stride) {
        float4 a0 = A4[i];
        float4 b0 = B4[i];
        float4 a1 = A4[i + (int)blockDim.x];
        float4 b1 = B4[i + (int)blockDim.x];
        float4 a2 = A4[i + 2 * (int)blockDim.x];
        float4 b2 = B4[i + 2 * (int)blockDim.x];
        float4 a3 = A4[i + 3 * (int)blockDim.x];
        float4 b3 = B4[i + 3 * (int)blockDim.x];

        s0 = fmaf(a0.x, b0.x, s0);
        s1 = fmaf(a0.y, b0.y, s1);
        s2 = fmaf(a0.z, b0.z, s2);
        s3 = fmaf(a0.w, b0.w, s3);

        s0 = fmaf(a1.x, b1.x, s0);
        s1 = fmaf(a1.y, b1.y, s1);
        s2 = fmaf(a1.z, b1.z, s2);
        s3 = fmaf(a1.w, b1.w, s3);

        s0 = fmaf(a2.x, b2.x, s0);
        s1 = fmaf(a2.y, b2.y, s1);
        s2 = fmaf(a2.z, b2.z, s2);
        s3 = fmaf(a2.w, b2.w, s3);

        s0 = fmaf(a3.x, b3.x, s0);
        s1 = fmaf(a3.y, b3.y, s1);
        s2 = fmaf(a3.z, b3.z, s2);
        s3 = fmaf(a3.w, b3.w, s3);
    }
    // Tail (for generality)
    for (; i < K4; i += (int)blockDim.x) {
        float4 a = A4[i];
        float4 b = B4[i];
        s0 = fmaf(a.x, b.x, s0);
        s1 = fmaf(a.y, b.y, s1);
        s2 = fmaf(a.z, b.z, s2);
        s3 = fmaf(a.w, b.w, s3);
    }

    float sum = (s0 + s1) + (s2 + s3);

    sum = wave_reduce_sum(sum);

    __shared__ float partial[4];
    if ((tid & (warpSize - 1)) == 0) {
        partial[tid / warpSize] = sum;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (tid < 4) block_sum = partial[tid];
    if (tid < warpSize) {
        block_sum = wave_reduce_sum(block_sum);
        if (tid == 0) {
            C[row] = block_sum;
        }
    }
}

torch::Tensor matvec_fp32_cuda(torch::Tensor A, torch::Tensor B) {
    if (!A.is_cuda() || !B.is_cuda()) {
        return at::matmul(A, B);
    }

    TORCH_CHECK(A.scalar_type() == at::kFloat, "A must be float32");
    TORCH_CHECK(B.scalar_type() == at::kFloat, "B must be float32");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(B.size(1) == 1, "B must be (K, 1)");
    TORCH_CHECK(A.size(1) == B.size(0), "K dimension mismatch");

    if (!A.is_contiguous()) A = A.contiguous();
    if (!B.is_contiguous()) B = B.contiguous();

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    TORCH_CHECK((K % 4) == 0, "K must be divisible by 4");
    const int K4 = (int)(K / 4);

    auto C = torch::empty({M, 1}, A.options());

    const int threads = 256;
    const dim3 blocks((unsigned int)M);

    auto stream = at::cuda::getDefaultCUDAStream().stream();
    gemv_fp32_vec4_kernel<<<blocks, threads, 0, stream>>>(
        (const float*)A.data_ptr<float>(),
        (const float*)B.data_ptr<float>(),
        (float*)C.data_ptr<float>(),
        K4);

    return C;
}
'''

_matvec_ext = load_inline(
    name="matvec_fp32_ext",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=["matvec_fp32"],
    with_cuda=True,
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _matvec_ext.matvec_fp32(A, B)


M = 256 * 8  # 2048
K = 131072 * 8  # 1048576


def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, 1)
    return [A, B]


def get_init_inputs():
    return []
