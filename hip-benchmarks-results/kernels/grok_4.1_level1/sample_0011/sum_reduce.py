import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>

__global__ void sum_reduce_kernel(
    const float *x, float *out, int64_t num_out,
    int64_t reduce_size, int64_t reduce_stride,
    int64_t size0, int64_t size1, int64_t size2,
    int64_t stride0, int64_t stride1, int64_t stride2,
    int dim
) {
    int out_idx = blockIdx.x;
    if (out_idx >= num_out) return;

    int tid = threadIdx.x;

    // correct row-major unravel for out_shape
    int temp = (int)out_idx;
    int coord_0, coord_1, coord_2;
    // dim 2 (innermost)
    if (dim != 2) {
        int sz_2 = (int)size2;
        coord_2 = temp % sz_2;
        temp /= sz_2;
    } else {
        coord_2 = 0;
    }
    // dim 1
    if (dim != 1) {
        int sz_1 = (int)size1;
        coord_1 = temp % sz_1;
        temp /= sz_1;
    } else {
        coord_1 = 0;
    }
    // dim 0
    if (dim != 0) {
        int sz_0 = (int)size0;
        coord_0 = temp % sz_0;
        temp /= sz_0;
    } else {
        coord_0 = 0;
    }

    // compute input base
    int64_t base = static_cast<int64_t>(coord_0) * stride0 +
                   static_cast<int64_t>(coord_1) * stride1 +
                   static_cast<int64_t>(coord_2) * stride2;

    // partial sum
    float partial = 0.0f;
    int bs = 256;
    for (int k = tid; k < (int)reduce_size; k += bs) {
        partial += x[base + static_cast<int64_t>(k) * reduce_stride];
    }

    // block reduce
    __shared__ float sdata[256];
    sdata[tid] = partial;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[out_idx] = sdata[0];
    }
}

__global__ void sum_reduce_dim1_serial(
    const float *x, float *out,
    int64_t B, int64_t N, int64_t M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * M) return;

    int b = idx / (int)M;
    int m = idx % (int)M;
    int64_t row_start = b * N * M;
    float sumv = 0.0f;
    for (int n = 0; n < (int)N; ++n) {
        sumv += x[row_start + (int64_t)n * M + m];
    }
    out[idx] = sumv;
}

torch::Tensor sum_reduce_hip(torch::Tensor x, int64_t dim_) {
    c10::IntArrayRef shape_ref = x.sizes();
    size_t rank = shape_ref.size();
    if (rank != 3) {
        return torch::sum(x, dim_, true);
    }
    auto shape_vec = shape_ref.vec();
    int64_t size0 = shape_vec[0], size1 = shape_vec[1], size2 = shape_vec[2];
    int64_t reduce_size = shape_vec[dim_];
    std::vector<int64_t> out_shape_vec = {size0, size1, size2};
    out_shape_vec[dim_] = 1;
    torch::Tensor out = torch::zeros(torch::IntArrayRef(out_shape_vec), x.options());

    int64_t num_out = out.numel();
    c10::IntArrayRef stride_ref = x.strides();
    auto x_strides_vec = stride_ref.vec();
    int64_t stride0 = x_strides_vec[0], stride1 = x_strides_vec[1], stride2 = x_strides_vec[2];
    int64_t reduce_stride = x_strides_vec[dim_];

    const int BS = 256;
    dim3 block(BS);

    if (dim_ == 1 && stride2 == 1 && stride1 == size2 && stride0 == size1 * size2) {
        // special fast serial coalesced for dim=1 contiguous
        dim3 grid(static_cast<unsigned int>((size0 * size2 + BS - 1) / BS));
        hipLaunchKernelGGL(sum_reduce_dim1_serial, grid, block, 0, 0,
                           x.data_ptr<float>(), out.data_ptr<float>(),
                           size0, size1, size2);
    } else {
        // general
        dim3 grid(static_cast<unsigned int>(num_out));
        hipLaunchKernelGGL(sum_reduce_kernel, grid, block, 0, 0,
                           x.data_ptr<float>(), out.data_ptr<float>(), num_out,
                           reduce_size, reduce_stride, size0, size1, size2,
                           stride0, stride1, stride2, static_cast<int>(dim_));
    }

    return out;
}
"""

sum_reduce = load_inline(
    name="sum_reduce",
    cpp_sources=cpp_source,
    functions=["sum_reduce_hip"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.sum_reduce = sum_reduce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sum_reduce.sum_reduce_hip(x, self.dim)

batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [reduce_dim]
