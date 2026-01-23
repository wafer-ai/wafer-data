import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Ensure we compile with hipcc on ROCm
os.environ.setdefault("CXX", "hipcc")

# Row-wise softmax for FP32, optimized for large feature dimension.
# One HIP block computes one row.
softmax_cpp_source = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <limits>

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__ 1
#endif

__global__ void softmax_row_fp32_kernel(const float* __restrict__ x,
                                       float* __restrict__ y,
                                       int rows, int cols) {
    // grid.x = row index
    int row = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    if (row >= rows) return;

    const float* row_x = x + ((int64_t)row) * cols;
    float* row_y = y + ((int64_t)row) * cols;

    // Pass 1: max
    float local_max = -INFINITY;

    int vec_cols = cols / 4;
    const float4* row_x4 = reinterpret_cast<const float4*>(row_x);

    for (int j = tid; j < vec_cols; j += (int)blockDim.x) {
        float4 v = row_x4[j];
        local_max = fmaxf(local_max, v.x);
        local_max = fmaxf(local_max, v.y);
        local_max = fmaxf(local_max, v.z);
        local_max = fmaxf(local_max, v.w);
    }

    // tail
    for (int j = vec_cols * 4 + tid; j < cols; j += (int)blockDim.x) {
        local_max = fmaxf(local_max, row_x[j]);
    }

    __shared__ float smax[256];
    smax[tid] = local_max;
    __syncthreads();

    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = smax[0];

    // Pass 2: sum exp(x - max)
    float local_sum = 0.0f;
    for (int j = tid; j < vec_cols; j += (int)blockDim.x) {
        float4 v = row_x4[j];
        local_sum += __expf(v.x - max_val);
        local_sum += __expf(v.y - max_val);
        local_sum += __expf(v.z - max_val);
        local_sum += __expf(v.w - max_val);
    }
    for (int j = vec_cols * 4 + tid; j < cols; j += (int)blockDim.x) {
        local_sum += __expf(row_x[j] - max_val);
    }

    __shared__ float ssum[256];
    ssum[tid] = local_sum;
    __syncthreads();

    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
        }
        __syncthreads();
    }
    float inv_sum = 1.0f / ssum[0];

    // Pass 3: write output
    for (int j = tid; j < vec_cols; j += (int)blockDim.x) {
        float4 v = row_x4[j];
        float4 o;
        o.x = __expf(v.x - max_val) * inv_sum;
        o.y = __expf(v.y - max_val) * inv_sum;
        o.z = __expf(v.z - max_val) * inv_sum;
        o.w = __expf(v.w - max_val) * inv_sum;
        reinterpret_cast<float4*>(row_y)[j] = o;
    }
    for (int j = vec_cols * 4 + tid; j < cols; j += (int)blockDim.x) {
        row_y[j] = __expf(row_x[j] - max_val) * inv_sum;
    }
}

torch::Tensor softmax_fp32_hip(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA/HIP tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [batch, features]");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    const auto rows = (int)x.size(0);
    const auto cols = (int)x.size(1);
    auto y = torch::empty_like(x);

    const int threads = 256;
    dim3 block(threads);
    dim3 grid(rows);

    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    softmax_row_fp32_kernel<<<grid, block, 0, stream>>>(
        (const float*)x.data_ptr<float>(),
        (float*)y.data_ptr<float>(),
        rows,
        cols);

    return y;
}
"""

softmax_ext = load_inline(
    name="softmax_rowwise_fp32_ext",
    cpp_sources=softmax_cpp_source,
    functions=["softmax_fp32_hip"],
    with_cuda=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """Optimized: keep rocBLAS GEMM (nn.Linear), keep Dropout semantics,
    replace softmax with a custom row-wise HIP kernel (FP32).
    """

    def __init__(self, in_features, out_features, dropout_p):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)
        self.softmax_ext = softmax_ext

    def forward(self, x):
        x = self.matmul(x)
        x = self.dropout(x)
        # Softmax over features (dim=1)
        return self.softmax_ext.softmax_fp32_hip(x)


# Keep the same IO helpers as the reference
batch_size = 128
in_features = 16384
out_features = 16384
dropout_p = 0.2

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, dropout_p]
