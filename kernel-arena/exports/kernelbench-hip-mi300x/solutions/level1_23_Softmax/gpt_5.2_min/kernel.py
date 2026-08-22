import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ.setdefault("CXX", "hipcc")

softmax_cpp_source = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>

__device__ __forceinline__ float block_reduce_max(float val, float* smem) {
    int tid = (int)threadIdx.x;
    smem[tid] = val;
    __syncthreads();
    for (int offset = (int)blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float other = smem[tid + offset];
            smem[tid] = other > smem[tid] ? other : smem[tid];
        }
        __syncthreads();
    }
    return smem[0];
}

__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    int tid = (int)threadIdx.x;
    smem[tid] = val;
    __syncthreads();
    for (int offset = (int)blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) smem[tid] += smem[tid + offset];
        __syncthreads();
    }
    return smem[0];
}

__global__ void softmax_row_fp32_large(const float* __restrict__ x, float* __restrict__ y, int dim) {
    extern __shared__ float smem[];
    int row = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    const float* row_x = x + (size_t)row * (size_t)dim;
    float* row_y = y + (size_t)row * (size_t)dim;

    float local_max = -INFINITY;
    for (int col = tid; col < dim; col += (int)blockDim.x) {
        float v = row_x[col];
        local_max = v > local_max ? v : local_max;
    }
    float max_val = block_reduce_max(local_max, smem);
    __syncthreads();

    float local_sum = 0.0f;
    for (int col = tid; col < dim; col += (int)blockDim.x) {
        float v = row_x[col] - max_val;
        local_sum += __expf(v);
    }
    float sum_val = block_reduce_sum(local_sum, smem);
    __syncthreads();

    float inv_sum = 1.0f / sum_val;
    for (int col = tid; col < dim; col += (int)blockDim.x) {
        float v = row_x[col] - max_val;
        row_y[col] = __expf(v) * inv_sum;
    }
}

torch::Tensor softmax_hip(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA/HIP tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");

    auto x_contig = x.contiguous();
    int64_t batch = x_contig.size(0);
    int64_t dim = x_contig.size(1);
    auto y = torch::empty_like(x_contig);

    int threads = 1024;
    if (dim < 1024) threads = 256;

    dim3 block(threads);
    dim3 grid((unsigned)batch);
    size_t shmem = (size_t)threads * sizeof(float);

    hipStream_t stream = at::hip::getDefaultHIPStream();
    hipLaunchKernelGGL(softmax_row_fp32_large, grid, block, shmem, stream,
                       (const float*)x_contig.data_ptr<float>(),
                       (float*)y.data_ptr<float>(),
                       (int)dim);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_hip", &softmax_hip, "Row-wise softmax FP32 (HIP)");
}
"""

# Use a unique extension name to avoid collisions in torch extension cache.
softmax_ext = load_inline(
    name="kb_softmax_row_fp32_ext_v2",
    cpp_sources=softmax_cpp_source,
    functions=None,
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return softmax_ext.softmax_hip(x)


def get_inputs():
    batch_size = 4096
    dim = 393216
    x = torch.rand(batch_size, dim, device="cuda", dtype=torch.float32)
    return [x]


def get_init_inputs():
    return []
