
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

layernorm_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

#define BLOCKS_PER_ROW 128
#define THREADS_PER_BLOCK 256

__global__ void partial_reduce_kernel(const float* __restrict__ x, double* __restrict__ partial_stats, int N) {
    int row_idx = blockIdx.x;
    int block_in_row = blockIdx.y;
    int tid = threadIdx.x;
    
    int elements_per_block = (N + BLOCKS_PER_ROW - 1) / BLOCKS_PER_ROW;
    int start_idx = row_idx * N + block_in_row * elements_per_block;
    int end_idx = min(start_idx + elements_per_block, (row_idx + 1) * N);
    
    double sum = 0.0;
    double sq_sum = 0.0;
    
    int i = start_idx + tid * 4;
    for (; i + 3 < end_idx; i += THREADS_PER_BLOCK * 4) {
        float4 val4 = reinterpret_cast<const float4*>(&x[i])[0];
        sum += (double)val4.x + (double)val4.y + (double)val4.z + (double)val4.w;
        sq_sum += (double)val4.x * val4.x + (double)val4.y * val4.y + (double)val4.z * val4.z + (double)val4.w * val4.w;
    }
    for (; i < end_idx; i++) {
        double val = (double)x[i];
        sum += val;
        sq_sum += val * val;
    }
    
    __shared__ double s_sum[THREADS_PER_BLOCK];
    __shared__ double s_sq_sum[THREADS_PER_BLOCK];
    s_sum[tid] = sum;
    s_sq_sum[tid] = sq_sum;
    __syncthreads();
    
    for (int s = THREADS_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sq_sum[tid] += s_sq_sum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_stats[(row_idx * BLOCKS_PER_ROW + block_in_row) * 2] = s_sum[0];
        partial_stats[(row_idx * BLOCKS_PER_ROW + block_in_row) * 2 + 1] = s_sq_sum[0];
    }
}

__global__ void layernorm_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const double* __restrict__ partial_stats,
    float* __restrict__ out,
    int N,
    double eps) {
    
    int row_idx = blockIdx.x;
    int block_in_row = blockIdx.y;
    int tid = threadIdx.x;
    
    __shared__ double s_mean;
    __shared__ double s_inv_std;
    
    if (tid == 0) {
        double total_sum = 0.0;
        double total_sq_sum = 0.0;
        for (int i = 0; i < BLOCKS_PER_ROW; ++i) {
            total_sum += partial_stats[(row_idx * BLOCKS_PER_ROW + i) * 2];
            total_sq_sum += partial_stats[(row_idx * BLOCKS_PER_ROW + i) * 2 + 1];
        }
        double mean = total_sum / N;
        double var = (total_sq_sum / N) - (mean * mean);
        if (var < 0) var = 0;
        s_mean = mean;
        s_inv_std = 1.0 / sqrt(var + eps);
    }
    __syncthreads();
    
    float mean = (float)s_mean;
    float inv_std = (float)s_inv_std;
    
    int elements_per_block = (N + BLOCKS_PER_ROW - 1) / BLOCKS_PER_ROW;
    int start_in_row = block_in_row * elements_per_block;
    int end_in_row = min(start_in_row + elements_per_block, N);
    
    int row_offset = row_idx * N;
    
    int i = start_in_row + tid * 4;
    for (; i + 3 < end_in_row; i += THREADS_PER_BLOCK * 4) {
        float4 x4 = reinterpret_cast<const float4*>(&x[row_offset + i])[0];
        float4 w4 = reinterpret_cast<const float4*>(&weight[i])[0];
        float4 b4 = reinterpret_cast<const float4*>(&bias[i])[0];
        
        float4 out4;
        out4.x = (x4.x - mean) * inv_std * w4.x + b4.x;
        out4.y = (x4.y - mean) * inv_std * w4.y + b4.y;
        out4.z = (x4.z - mean) * inv_std * w4.z + b4.z;
        out4.w = (x4.w - mean) * inv_std * w4.w + b4.w;
        
        reinterpret_cast<float4*>(&out[row_offset + i])[0] = out4;
    }
    for (; i < end_in_row; i++) {
        out[row_offset + i] = (x[row_offset + i] - mean) * inv_std * weight[i] + bias[i];
    }
}

torch::Tensor layernorm_hip(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps) {
    auto input_shape = x.sizes();
    int num_batches = input_shape[0];
    int N = 1;
    for (int i = 1; i < input_shape.size(); ++i) {
        N *= input_shape[i];
    }

    auto out = torch::empty_like(x);
    auto partial_stats = torch::empty({num_batches, BLOCKS_PER_ROW, 2}, x.options().dtype(torch::kFloat64));

    dim3 grid(num_batches, BLOCKS_PER_ROW);
    dim3 block(THREADS_PER_BLOCK);
    
    partial_reduce_kernel<<<grid, block>>>(x.data_ptr<float>(), partial_stats.data_ptr<double>(), N);
    layernorm_norm_kernel<<<grid, block>>>(x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), partial_stats.data_ptr<double>(), out.data_ptr<float>(), N, eps);

    return out;
}
"""

layernorm_module = load_inline(
    name="layernorm_hip_v2",
    cpp_sources="""
    torch::Tensor layernorm_hip(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps);
    """,
    cuda_sources=layernorm_hip_source,
    functions=["layernorm_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return layernorm_module.layernorm_hip(x, self.weight.view(-1), self.bias.view(-1), self.eps)
