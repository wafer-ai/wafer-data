import os
os.environ["CXX"] = "hipcc"

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softmax_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__device__ float warp_reduce_max(float val) {
    uint64_t mask = 0xffffffffffffffffUL;
    val = fmaxf(val, __shfl_xor_sync(mask, val, 32, 64));
    val = fmaxf(val, __shfl_xor_sync(mask, val, 16, 64));
    val = fmaxf(val, __shfl_xor_sync(mask, val, 8, 64));
    val = fmaxf(val, __shfl_xor_sync(mask, val, 4, 64));
    val = fmaxf(val, __shfl_xor_sync(mask, val, 2, 64));
    val = fmaxf(val, __shfl_xor_sync(mask, val, 1, 64));
    return val;
}

__device__ float warp_reduce_sum(float val) {
    uint64_t mask = 0xffffffffffffffffUL;
    val += __shfl_xor_sync(mask, val, 32, 64);
    val += __shfl_xor_sync(mask, val, 16, 64);
    val += __shfl_xor_sync(mask, val, 8, 64);
    val += __shfl_xor_sync(mask, val, 4, 64);
    val += __shfl_xor_sync(mask, val, 2, 64);
    val += __shfl_xor_sync(mask, val, 1, 64);
    return val;
}

__global__ void row_max_kernel(const float *x, float *maxes, int B, int D) {
    int row = blockIdx.x;
    if (row >= B) return;
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bs = blockDim.x;
    int lane = tid % 64;
    int wid = tid / 64;
    float lmax = -1e30f;
    int offset = row * D;
    for (int i = tid; i < D; i += bs) {
        float v = x[offset + i];
        lmax = fmaxf(lmax, v);
    }
    float wmax = warp_reduce_max(lmax);
    if (lane == 0) {
        sdata[wid] = wmax;
    }
    __syncthreads();
    int num_w = bs / 64;
    if (lane == 0) {
        for (int s = num_w / 2; s > 0; s >>= 1) {
            if (wid < s) {
                sdata[wid] = fmaxf(sdata[wid], sdata[wid + s]);
            }
            __syncthreads();
        }
        if (wid == 0) {
            maxes[row] = sdata[0];
        }
    }
}

__global__ void row_exp_sum_kernel(const float *x, const float *maxes, float *sums, int B, int D) {
    int row = blockIdx.x;
    if (row >= B) return;
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bs = blockDim.x;
    int lane = tid % 64;
    int wid = tid / 64;
    float lsum = 0.0f;
    float mmax = maxes[row];
    int offset = row * D;
    for (int i = tid; i < D; i += bs) {
        float v = x[offset + i] - mmax;
        lsum += (v < -30.0f) ? 0.0f : expf(v);
    }
    float wsum = warp_reduce_sum(lsum);
    if (lane == 0) {
        sdata[wid] = wsum;
    }
    __syncthreads();
    int num_w = bs / 64;
    if (lane == 0) {
        for (int s = num_w / 2; s > 0; s >>= 1) {
            if (wid < s) {
                sdata[wid] += sdata[wid + s];
            }
            __syncthreads();
        }
        if (wid == 0) {
            sums[row] = sdata[0];
        }
    }
}

__global__ void row_scale_kernel(const float *x, const float *maxes, const float *sums, float *out, int B, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * D) return;
    int row = idx / D;
    float mmax = maxes[row];
    float v = x[idx] - mmax;
    out[idx] = ((v < -30.0f) ? 0.0f : expf(v)) / sums[row];
}

torch::Tensor softmax_hip(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    auto num_rows_long = x.size(0);
    auto seq_len_long = x.size(1);
    TORCH_CHECK(num_rows_long <= 2147483647LL && seq_len_long <= 2147483647LL, "Dimensions too large");
    int B = static_cast<int>(num_rows_long);
    int D = static_cast<int>(seq_len_long);
    auto out = torch::empty_like(x);
    auto options = x.options();
    auto row_max = torch::empty({B}, options);
    auto row_sums = torch::empty({B}, options);
    const int bs_reduce = 256;
    const int bs_scale = 1024;
    dim3 block_reduce(bs_reduce);
    dim3 grid(B);
    size_t shmem_size = (bs_reduce / 64) * sizeof(float);
    hipLaunchKernelGGL(row_max_kernel, grid, block_reduce, shmem_size, 0, x.data_ptr<float>(), row_max.data_ptr<float>(), B, D);
    hipLaunchKernelGGL(row_exp_sum_kernel, grid, block_reduce, shmem_size, 0, x.data_ptr<float>(), row_max.data_ptr<float>(), row_sums.data_ptr<float>(), B, D);
    long long total = (long long)B * D;
    dim3 block_scale(bs_scale);
    dim3 grid_scale(static_cast<unsigned int>((total + bs_scale - 1LL) / bs_scale));
    hipLaunchKernelGGL(row_scale_kernel, grid_scale, block_scale, 0, 0, x.data_ptr<float>(), row_max.data_ptr<float>(), row_sums.data_ptr<float>(), out.data_ptr<float>(), B, D);
    return out;
}
"""

softmax_module = load_inline(
    name="softmax",
    cpp_sources=softmax_cpp_source,
    functions=["softmax_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softmax_hip = softmax_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax_hip.softmax_hip(x)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
