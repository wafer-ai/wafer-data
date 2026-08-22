import os
os.environ["CXX"] = "hipcc"
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp = """
#include <hip/hip_runtime.h>

__global__ void fused_pool_sum_scale_kernel(const float *y, float scale, float *out, int B, int M, int kernel_size) {
    int b = blockIdx.x;
    if (b >= B) return;
    const float *yb = y + b * M;
    int num_pools = M / kernel_size;
    double local_sum = 0.0;
    int tid = threadIdx.x;
    int step = blockDim.x;
    for (int p = tid; p < num_pools; p += step) {
        int start = p * kernel_size;
        float maxv = yb[start];
        #pragma unroll
        for (int k = 1; k < kernel_size; ++k) {
            maxv = fmaxf(maxv, yb[start + k]);
        }
        local_sum += static_cast<double>(maxv);
    }
    extern __shared__ double sdata[];
    sdata[tid] = local_sum;
    __syncthreads();
    int offset = blockDim.x >> 1;
    for (; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[b] = sdata[0] * scale;
    }
}

torch::Tensor fused_pool_sum_scale_hip(torch::Tensor y, torch::Tensor scale_t, torch::Tensor ks_t) {
    auto B_ = y.size(0);
    auto M_ = y.size(1);
    int B = static_cast<int>(B_);
    int M = static_cast<int>(M_);
    float scale = *scale_t.data_ptr<float>();
    int kernel_size = *ks_t.data_ptr<int>();
    auto out = torch::empty({B_}, y.options());
    const int block_size = 512;
    const int grid_size = B;
    size_t shmem_bytes = block_size * sizeof(double);
    dim3 grid(grid_size);
    dim3 blk(block_size);
    hipLaunchKernelGGL(fused_pool_sum_scale_kernel, grid, blk, shmem_bytes, 0, y.data_ptr<float>(), scale, out.data_ptr<float>(), B, M, kernel_size);
    return out;
}
"""

fused_post = load_inline(
    name="fused_post",
    cpp_sources=cpp,
    functions=["fused_pool_sum_scale_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.fused_post = fused_post

    def forward(self, x):
        y = self.matmul(x)
        device = y.device
        dtype = y.dtype
        scale_t = torch.tensor(self.scale_factor, dtype=dtype, device=device)
        ks_t = torch.tensor(self.kernel_size, dtype=torch.int32, device=device)
        out = self.fused_post.fused_pool_sum_scale_hip(y, scale_t, ks_t)
        return out

def get_inputs():
    return [torch.rand(128, 32768)]

def get_init_inputs():
    return [32768, 32768, 2, 0.5]