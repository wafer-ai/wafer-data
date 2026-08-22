import os
os.environ["CXX"] = "hipcc"
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

instancenorm_cpp = """
#include <hip/hip_runtime.h>
#include <cmath>

__global__ void moments_kernel(const float* x, float* sumbuf, float* sumsqbuf, int C, int H, int W) {
    int nc_id = blockIdx.x;
    size_t HW = (size_t)H * W;
    size_t base = (size_t)nc_id * HW;
    const int tpb = 256;
    __shared__ float sh_sum[256];
    __shared__ float sh_sumsq[256];
    int tid = threadIdx.x;
    float lsum = 0.0f;
    float lsumsq = 0.0f;
    for (size_t i = tid; i < HW; i += tpb) {
        float v = x[base + i];
        lsum += v;
        lsumsq += v * v;
    }
    sh_sum[tid] = lsum;
    sh_sumsq[tid] = lsumsq;
    __syncthreads();
    for (int d = 128; d > 0; d >>= 1) {
        if (tid < d) {
            sh_sum[tid] += sh_sum[tid + d];
            sh_sumsq[tid] += sh_sumsq[tid + d];
        }
        __syncthreads();
    }
    if (tid == 0) {
        sumbuf[nc_id] = sh_sum[0];
        sumsqbuf[nc_id] = sh_sumsq[0];
    }
}

__global__ void compute_stats_kernel(const float* sums, const float* sumsq, float* means, float* invstds, float hw_inv, float eps, int nc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nc) return;
    float s = sums[tid];
    float ss = sumsq[tid];
    float mean = s * hw_inv;
    means[tid] = mean;
    float var = ss * hw_inv - mean * mean;
    invstds[tid] = 1.0f / sqrtf(var + eps);
}

__global__ void norm_kernel(const float* x, const float* means, const float* invstds, float* y, float scale, size_t total_size, size_t hw) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    size_t nc_id = idx / hw;
    float diff = x[idx] - means[nc_id];
    y[idx] = diff * invstds[nc_id] * scale;
}

torch::Tensor instancenorm_divide_hip(torch::Tensor input_, float divide_by_) {
    float scale = 1.0f / divide_by_;
    auto input = input_.contiguous();
    auto sizes = input.sizes();
    int64_t n = sizes[0];
    int64_t c = sizes[1];
    int64_t h = sizes[2];
    int64_t w = sizes[3];
    int64_t nc_ = n * c;
    int64_t hw_ = h * w;
    int64_t total_ = nc_ * hw_;
    if (total_ == 0) return torch::zeros_like(input);
    auto opts = input.options();
    auto sums_ = torch::zeros({nc_}, opts);
    auto sumsq_ = torch::zeros({nc_}, opts);
    auto means_ = torch::zeros({nc_}, opts);
    auto invstds_ = torch::zeros({nc_}, opts);
    auto output = torch::empty_like(input);
    float* xptr = input.data_ptr<float>();
    float* sumsptr = sums_.data_ptr<float>();
    float* sumsqptr = sumsq_.data_ptr<float>();
    int nci = (int)nc_;
    int ci = (int)c;
    int hi = (int)h;
    int wi = (int)w;
    dim3 grid_mom(nci);
    dim3 block_mom(256);
    moments_kernel<<<grid_mom, block_mom>>>(xptr, sumsptr, sumsqptr, ci, hi, wi);
    float hw_inv_f = 1.0f / (float)hw_;
    float eps_f = 1e-5f;
    dim3 grid_stat((nci + 255) / 256);
    dim3 block_stat(256);
    compute_stats_kernel<<<grid_stat, block_stat>>>(sumsptr, sumsqptr, means_.data_ptr<float>(), invstds_.data_ptr<float>(), hw_inv_f, eps_f, nci);
    dim3 grid_norm((total_ + 1023) / 1024);
    dim3 block_norm(1024);
    norm_kernel<<<grid_norm, block_norm>>>(xptr, means_.data_ptr<float>(), invstds_.data_ptr<float>(), output.data_ptr<float>(), scale, (size_t)total_, (size_t)hw_);
    return output;
}
"""

norm = load_inline(
    name="instancenorm_divide",
    cpp_sources=instancenorm_cpp,
    functions=["instancenorm_divide_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divide_by = divide_by

    def forward(self, x):
        x = self.conv(x)
        x = norm.instancenorm_divide_hip(x, float(self.divide_by))
        return x

batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
divide_by = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divide_by]
