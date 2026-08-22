import os
os.environ["CXX"] = "hipcc"

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

layernorm_cpp_source = """
#include "torch/extension.h"
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <cmath>

__global__ void compute_sums_kernel(const float* x, float* sumx, float* sumx2, int stride_b, int vol) {
    int b = blockIdx.z;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    extern __shared__ float sdata[];
    float* s_sumx = sdata;
    float* s_sumx2 = sdata + block_size;
    size_t block_offset = static_cast<size_t>(bid) * block_size;
    size_t global_tid = block_offset + tid;
    float val = 0.0f;
    if (global_tid < static_cast<size_t>(vol)) {
        const float* ptr_b = x + static_cast<size_t>(b) * stride_b;
        val = ptr_b[global_tid];
    }
    s_sumx[tid] = val;
    s_sumx2[tid] = val * val;
    __syncthreads();
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sumx[tid] += s_sumx[tid + s];
            s_sumx2[tid] += s_sumx2[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(sumx + b, s_sumx[0]);
        atomicAdd(sumx2 + b, s_sumx2[0]);
    }
}

__global__ void layernorm_apply_kernel(const float* x, const float* means, const float* inv_vars, const float* weight, const float* bias, float* y, int stride_b, int vol, int total_nelem) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_nelem) return;
    int b = tid / vol;
    int offset = tid % vol;
    size_t out_idx = static_cast<size_t>(b) * stride_b + offset;
    float val = x[out_idx];
    float m = means[b];
    float iv = inv_vars[b];
    y[out_idx] = (val - m) * iv * weight[offset] + bias[offset];
}

torch::Tensor layernorm_hip(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float eps) {
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    auto sizes = x.sizes();
    int64_t B = sizes[0];
    int64_t C = sizes[1];
    int64_t H = sizes[2];
    int64_t W = sizes[3];
    int64_t vol = C * H * W;
    int64_t stride_b = vol;
    TORCH_CHECK(weight.numel() == vol, "weight size mismatch");
    TORCH_CHECK(bias.numel() == vol, "bias size mismatch");
    auto opts = x.options();
    auto sumx = torch::zeros({B}, opts);
    auto sumx2 = torch::zeros({B}, opts);
    auto means_t = torch::zeros({B}, opts);
    auto inv_vars = torch::zeros({B}, opts);
    auto out = torch::empty_like(x);
    const int block_size = 1024;
    int64_t blocks_per_vol = (vol + block_size - 1LL) / block_size;
    dim3 grid_dims(static_cast<unsigned int>(blocks_per_vol), 1u, static_cast<unsigned int>(B));
    dim3 block_dims(block_size, 1u, 1u);
    size_t shmem_size = 2ULL * block_size * sizeof(float);
    hipLaunchKernelGGL(compute_sums_kernel, grid_dims, block_dims, shmem_size, 0,
                       x.data_ptr<float>(), sumx.data_ptr<float>(), sumx2.data_ptr<float>(),
                       static_cast<int>(stride_b), static_cast<int>(vol));
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("compute_sums_kernel error: ") + hipGetErrorString(err));
    }
    hipDeviceSynchronize();
    auto sumx_a = sumx.accessor<float, 1>();
    auto sumx2_a = sumx2.accessor<float, 1>();
    auto means_a = means_t.accessor<float, 1>();
    auto inv_vars_a = inv_vars.accessor<float, 1>();
    for (int64_t b = 0; b < B; ++b) {
        float sx = sumx_a[b];
        float sx2 = sumx2_a[b];
        float mean_val = sx / static_cast<float>(vol);
        float var_val = sx2 / static_cast<float>(vol) - mean_val * mean_val;
        inv_vars_a[b] = 1.0f / sqrtf(var_val + eps);
        means_a[b] = mean_val;
    }
    int64_t total_nelem = B * vol;
    int64_t num_blocks_apply = (total_nelem + block_size - 1LL) / block_size;
    hipLaunchKernelGGL(layernorm_apply_kernel, dim3(static_cast<unsigned int>(num_blocks_apply), 1u, 1u), dim3(block_size), 0, 0,
                       x.data_ptr<float>(), means_t.data_ptr<float>(), inv_vars.data_ptr<float>(),
                       weight.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(),
                       static_cast<int>(stride_b), static_cast<int>(vol), static_cast<int>(total_nelem));
    err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("apply_kernel error: ") + hipGetErrorString(err));
    }
    hipDeviceSynchronize();
    return out;
}
""";

layernorm = load_inline(
    name="layernorm",
    cpp_sources=layernorm_cpp_source,
    functions=["layernorm_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model with custom HIP kernel for LayerNorm.
    """
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.layernorm_hip = layernorm.layernorm_hip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layernorm_hip(x, self.weight, self.bias, self.eps)
