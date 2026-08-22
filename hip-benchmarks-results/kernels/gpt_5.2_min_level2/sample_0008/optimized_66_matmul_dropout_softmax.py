import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ.setdefault("CXX", "hipcc")

softmax_cpp = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>

static inline __device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) val = fmaxf(val, __shfl_down(val, offset));
    return val;
}
static inline __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) val += __shfl_down(val, offset);
    return val;
}

// In-place row-wise softmax, one block per row.
__global__ void softmax_row_inplace_fp32_kernel(float* __restrict__ x,
                                               int64_t rows,
                                               int64_t cols) {
    int row = (int)blockIdx.x;
    if (row >= rows) return;
    int tid = (int)threadIdx.x;
    int64_t base = (int64_t)row * cols;

    float local_max = -INFINITY;
    for (int64_t c = tid; c < cols; c += (int)blockDim.x) {
        float v = x[base + c];
        local_max = fmaxf(local_max, v);
    }
    local_max = warp_reduce_max(local_max);

    // 512 threads -> 16 warps
    __shared__ float s_max[16];
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) s_max[warp] = local_max;
    __syncthreads();

    if (warp == 0) {
        float v = (lane < (blockDim.x >> 5)) ? s_max[lane] : -INFINITY;
        v = warp_reduce_max(v);
        if (lane == 0) s_max[0] = v;
    }
    __syncthreads();
    float row_max = s_max[0];

    float local_sum = 0.0f;
    for (int64_t c = tid; c < cols; c += (int)blockDim.x) {
        float v = x[base + c] - row_max;
        local_sum += __expf(v);
    }
    local_sum = warp_reduce_sum(local_sum);

    __shared__ float s_sum[16];
    if (lane == 0) s_sum[warp] = local_sum;
    __syncthreads();

    if (warp == 0) {
        float v = (lane < (blockDim.x >> 5)) ? s_sum[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) s_sum[0] = v;
    }
    __syncthreads();
    float inv = 1.0f / s_sum[0];

    for (int64_t c = tid; c < cols; c += (int)blockDim.x) {
        float v = x[base + c] - row_max;
        x[base + c] = __expf(v) * inv;
    }
}

torch::Tensor softmax_row_fp32_inplace(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA/ROCm tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D (B, N)");

    auto x_contig = x.contiguous();
    const auto rows = x_contig.size(0);
    const auto cols = x_contig.size(1);

    const int threads = 512;
    dim3 blocks((unsigned int)rows);

    hipStream_t stream = (hipStream_t)at::cuda::getDefaultCUDAStream().stream();

    hipLaunchKernelGGL(softmax_row_inplace_fp32_kernel,
                      blocks,
                      dim3(threads),
                      0,
                      stream,
                      (float*)x_contig.data_ptr<float>(),
                      (int64_t)rows,
                      (int64_t)cols);

    return x_contig;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_row_fp32_inplace", &softmax_row_fp32_inplace, "In-place row-wise softmax FP32 (ROCm)");
}
"""

softmax_ext = load_inline(
    name="softmax_row_fp32_ext",
    cpp_sources=softmax_cpp,
    functions=None,
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.matmul(x)
        if self.training:
            x = self.dropout(x)
            return torch.softmax(x, dim=1)
        # eval: dropout is identity; do in-place softmax on matmul output
        return softmax_ext.softmax_row_fp32_inplace(x)


batch_size = 128
in_features = 16384
out_features = 16384
dropout_p = 0.2


def get_inputs():
    return [torch.rand(batch_size, in_features, device="cuda", dtype=torch.float32)]


def get_init_inputs():
    return [in_features, out_features, dropout_p]
