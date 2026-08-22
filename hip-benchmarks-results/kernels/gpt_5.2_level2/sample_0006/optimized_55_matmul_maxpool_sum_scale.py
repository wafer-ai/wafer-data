import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ.setdefault("CXX", "hipcc")

# Optimizations:
# - Keep the Linear GEMM on rocBLAS (already near peak on MI300X)
# - Replace MaxPool1d(kernel_size=2, stride=2) + sum + scale with a custom HIP implementation
# - Use a 2-stage reduction for kernel_size=2 to increase parallelism
# - Avoid per-forward temporary allocations by caching a partials buffer in the module

fused_cpp_source = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA/HIP tensor")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

static inline __device__ float wave_reduce_sum(float v) {
    for (int offset = 32; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset);
    }
    return v;
}

template<int NUM_WAVES>
static inline __device__ float block_reduce_sum(float v) {
    v = wave_reduce_sum(v);
    __shared__ float smem[NUM_WAVES];
    int tid = (int)threadIdx.x;
    int lane = tid & 63;
    int wave = tid >> 6;
    if (lane == 0) smem[wave] = v;
    __syncthreads();
    float sum = 0.0f;
    if (wave == 0) {
        sum = (lane < NUM_WAVES) ? smem[lane] : 0.0f;
        sum = wave_reduce_sum(sum);
    }
    return sum;
}

// Tuned for N=32768: N/4=8192 float4 per row.
// CHUNK_F4=256 => chunks=32, blocks=B*32.
constexpr int CHUNK_F4 = 256;

__global__ void k2_partial_sum_f4_kernel(const float* __restrict__ x,
                                        float* __restrict__ partial,
                                        int64_t N,
                                        int chunks) {
    int row = (int)blockIdx.x;
    int chunk = (int)blockIdx.y;

    const float4* row4 = reinterpret_cast<const float4*>(x + (int64_t)row * N);
    int64_t start = (int64_t)chunk * CHUNK_F4;

    float acc = 0.0f;
    int tid = (int)threadIdx.x;

    #pragma unroll 4
    for (int64_t i = tid; i < CHUNK_F4; i += (int64_t)blockDim.x) {
        float4 v = row4[start + i];
        acc += fmaxf(v.x, v.y) + fmaxf(v.z, v.w);
    }

    float sum = block_reduce_sum<4>(acc); // 256 threads => 4 waves
    if (tid == 0) partial[(int64_t)row * chunks + chunk] = sum;
}

__global__ void reduce_partials_scale_kernel(const float* __restrict__ partial,
                                            float* __restrict__ out,
                                            int chunks,
                                            float scale) {
    int row = (int)blockIdx.x;
    float acc = 0.0f;
    int tid = (int)threadIdx.x;

    const float* rowp = partial + (int64_t)row * chunks;
    for (int i = tid; i < chunks; i += (int)blockDim.x) {
        acc += rowp[i];
    }

    float sum = block_reduce_sum<4>(acc);
    if (tid == 0) out[row] = sum * scale;
}

// Fallback: single-block per row for generic kernel_size<=16
__global__ void maxpool_sum_scale_generic_kernel(const float* __restrict__ x,
                                                 float* __restrict__ out,
                                                 int64_t N,
                                                 int k,
                                                 float scale) {
    int row = (int)blockIdx.x;
    const float* row_ptr = x + (int64_t)row * N;
    int64_t out_len = N / k;

    float acc = 0.0f;
    int tid = (int)threadIdx.x;
    for (int64_t i = tid; i < out_len; i += (int64_t)blockDim.x) {
        int64_t base = i * k;
        float m = row_ptr[base];
        #pragma unroll
        for (int t = 1; t < 16; t++) {
            if (t < k) m = fmaxf(m, row_ptr[base + t]);
        }
        acc += m;
    }

    float sum = block_reduce_sum<4>(acc);
    if (tid == 0) out[row] = sum * scale;
}

// Signature includes a caller-provided partial buffer to avoid allocation in the hot path.
torch::Tensor maxpool_sum_scale_hip(torch::Tensor x,
                                   torch::Tensor partial,
                                   double scale_factor,
                                   int64_t kernel_size) {
    CHECK_CUDA(x);
    CHECK_FLOAT(x);
    CHECK_CONTIG(x);
    TORCH_CHECK(x.dim() == 2, "x must be 2D [B, N]");
    TORCH_CHECK(kernel_size > 0, "kernel_size must be > 0");

    int64_t B = x.size(0);
    int64_t N = x.size(1);

    auto out = torch::empty({B}, x.options());

    hipStream_t stream = at::hip::getDefaultHIPStream();
    float scale = (float)scale_factor;

    if (kernel_size == 2 && (N % 1024 == 0)) {
        // N divisible by 4 and (N/4) divisible by CHUNK_F4.
        int chunks = (int)(N / (4 * CHUNK_F4)); // N/1024 for CHUNK_F4=256
        TORCH_CHECK(chunks > 0, "chunks must be > 0");

        CHECK_CUDA(partial);
        CHECK_FLOAT(partial);
        CHECK_CONTIG(partial);
        TORCH_CHECK(partial.dim() == 2, "partial must be 2D [B, chunks]");
        TORCH_CHECK(partial.size(0) == B, "partial.size(0) must match B");
        TORCH_CHECK(partial.size(1) == chunks, "partial.size(1) must match expected chunks");

        dim3 block1(256);
        dim3 grid1((unsigned)B, (unsigned)chunks);
        hipLaunchKernelGGL(k2_partial_sum_f4_kernel, grid1, block1, 0, stream,
                           (const float*)x.data_ptr<float>(),
                           (float*)partial.data_ptr<float>(),
                           N,
                           chunks);

        dim3 block2(256);
        dim3 grid2((unsigned)B);
        hipLaunchKernelGGL(reduce_partials_scale_kernel, grid2, block2, 0, stream,
                           (const float*)partial.data_ptr<float>(),
                           (float*)out.data_ptr<float>(),
                           chunks,
                           scale);
    } else {
        TORCH_CHECK(kernel_size <= 16, "generic kernel supports kernel_size<=16");
        dim3 block(256);
        dim3 grid((unsigned)B);
        hipLaunchKernelGGL(maxpool_sum_scale_generic_kernel, grid, block, 0, stream,
                           (const float*)x.data_ptr<float>(),
                           (float*)out.data_ptr<float>(),
                           N,
                           (int)kernel_size,
                           scale);
    }

    auto err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(err));
    return out;
}
"""

fused_mod = load_inline(
    name="fused_maxpool_sum_scale_ext",
    cpp_sources=fused_cpp_source,
    functions=["maxpool_sum_scale_hip"],
    extra_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.kernel_size = int(kernel_size)
        self.scale_factor = float(scale_factor)
        self.fused = fused_mod
        self._partial = None

    def _get_partial(self, x: torch.Tensor) -> torch.Tensor:
        # Fast path expects N divisible by 1024 and kernel_size=2
        B, N = x.shape
        if self.kernel_size == 2 and (N % 1024 == 0):
            chunks = N // 1024
            if (
                self._partial is None
                or self._partial.device != x.device
                or self._partial.dtype != x.dtype
                or self._partial.shape != (B, chunks)
            ):
                self._partial = torch.empty((B, chunks), device=x.device, dtype=x.dtype)
            return self._partial
        # Not used for fallback, but must be a valid tensor argument
        if self._partial is None or self._partial.device != x.device or self._partial.dtype != x.dtype:
            self._partial = torch.empty((1, 1), device=x.device, dtype=x.dtype)
        return self._partial

    def forward(self, x):
        x = self.matmul(x)
        partial = self._get_partial(x)
        return self.fused.maxpool_sum_scale_hip(x, partial, self.scale_factor, self.kernel_size)


batch_size = 128
in_features = 32768
out_features = 32768
kernel_size = 2
scale_factor = 0.5

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]
