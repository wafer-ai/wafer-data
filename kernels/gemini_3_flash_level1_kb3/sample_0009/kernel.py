
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

layernorm_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define WAVEFRONT_SIZE 64

__global__ void layernorm_forward_kernel1_float4(
    const float4* __restrict__ input,
    double* __restrict__ partial_sums,
    double* __restrict__ partial_sq_sums,
    int64_t N_over_4,
    int num_blocks_per_group) {
    
    int group_idx = blockIdx.y;
    int block_in_group = blockIdx.x;
    int threads_per_block = blockDim.x;
    
    int64_t start_idx = (int64_t)group_idx * N_over_4;
    int64_t block_stride = (int64_t)threads_per_block * num_blocks_per_group;
    
    double sum = 0;
    double sq_sum = 0;
    
    for (int64_t i = (int64_t)block_in_group * threads_per_block + threadIdx.x; i < N_over_4; i += block_stride) {
        float4 val4 = input[start_idx + i];
        sum += (double)val4.x + (double)val4.y + (double)val4.z + (double)val4.w;
        sq_sum += (double)val4.x * (double)val4.x + (double)val4.y * (double)val4.y + 
                  (double)val4.z * (double)val4.z + (double)val4.w * (double)val4.w;
    }
    
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_xor(sum, offset, WAVEFRONT_SIZE);
        sq_sum += __shfl_xor(sq_sum, offset, WAVEFRONT_SIZE);
    }
    
    __shared__ double s_sum[1024 / WAVEFRONT_SIZE];
    __shared__ double s_sq_sum[1024 / WAVEFRONT_SIZE];
    
    int lane = threadIdx.x % WAVEFRONT_SIZE;
    int wid = threadIdx.x / WAVEFRONT_SIZE;
    
    if (lane == 0) {
        s_sum[wid] = sum;
        s_sq_sum[wid] = sq_sum;
    }
    
    __syncthreads();
    
    if (wid == 0) {
        sum = (lane < (threads_per_block / WAVEFRONT_SIZE)) ? s_sum[lane] : 0.0;
        sq_sum = (lane < (threads_per_block / WAVEFRONT_SIZE)) ? s_sq_sum[lane] : 0.0;
        
        for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_xor(sum, offset, WAVEFRONT_SIZE);
            sq_sum += __shfl_xor(sq_sum, offset, WAVEFRONT_SIZE);
        }
        
        if (lane == 0) {
            partial_sums[group_idx * num_blocks_per_group + block_in_group] = sum;
            partial_sq_sums[group_idx * num_blocks_per_group + block_in_group] = sq_sum;
        }
    }
}

__global__ void layernorm_forward_kernel1_basic(
    const float* __restrict__ input,
    double* __restrict__ partial_sums,
    double* __restrict__ partial_sq_sums,
    int64_t N,
    int num_blocks_per_group) {
    
    int group_idx = blockIdx.y;
    int block_in_group = blockIdx.x;
    int threads_per_block = blockDim.x;
    
    int64_t start_idx = (int64_t)group_idx * N;
    int64_t block_stride = (int64_t)threads_per_block * num_blocks_per_group;
    
    double sum = 0;
    double sq_sum = 0;
    
    for (int64_t i = (int64_t)block_in_group * threads_per_block + threadIdx.x; i < N; i += block_stride) {
        float val = input[start_idx + i];
        sum += (double)val;
        sq_sum += (double)val * (double)val;
    }
    
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_xor(sum, offset, WAVEFRONT_SIZE);
        sq_sum += __shfl_xor(sq_sum, offset, WAVEFRONT_SIZE);
    }
    
    __shared__ double s_sum[1024 / WAVEFRONT_SIZE];
    __shared__ double s_sq_sum[1024 / WAVEFRONT_SIZE];
    
    int lane = threadIdx.x % WAVEFRONT_SIZE;
    int wid = threadIdx.x / WAVEFRONT_SIZE;
    
    if (lane == 0) {
        s_sum[wid] = sum;
        s_sq_sum[wid] = sq_sum;
    }
    
    __syncthreads();
    
    if (wid == 0) {
        sum = (lane < (threads_per_block / WAVEFRONT_SIZE)) ? s_sum[lane] : 0.0;
        sq_sum = (lane < (threads_per_block / WAVEFRONT_SIZE)) ? s_sq_sum[lane] : 0.0;
        
        for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_xor(sum, offset, WAVEFRONT_SIZE);
            sq_sum += __shfl_xor(sq_sum, offset, WAVEFRONT_SIZE);
        }
        
        if (lane == 0) {
            partial_sums[group_idx * num_blocks_per_group + block_in_group] = sum;
            partial_sq_sums[group_idx * num_blocks_per_group + block_in_group] = sq_sum;
        }
    }
}

__global__ void layernorm_forward_kernel2(
    double* __restrict__ partial_sums,
    double* __restrict__ partial_sq_sums,
    float* __restrict__ means,
    float* __restrict__ inv_stds,
    int64_t N,
    int num_blocks_per_group,
    float eps) {
    
    int group_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    __shared__ double s_sum[1024];
    __shared__ double s_sq_sum[1024];
    
    double sum = 0;
    double sq_sum = 0;
    
    for (int i = tid; i < num_blocks_per_group; i += blockDim.x) {
        sum += partial_sums[group_idx * num_blocks_per_group + i];
        sq_sum += partial_sq_sums[group_idx * num_blocks_per_group + i];
    }
    
    s_sum[tid] = sum;
    s_sq_sum[tid] = sq_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sq_sum[tid] += s_sq_sum[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        double mean = s_sum[0] / (double)N;
        double var = (s_sq_sum[0] / (double)N) - (mean * mean);
        if (var < 0) var = 0;
        means[group_idx] = (float)mean;
        inv_stds[group_idx] = (float)(1.0 / sqrt(var + (double)eps));
    }
}

__global__ void layernorm_forward_kernel3_float4(
    const float4* __restrict__ input,
    const float* __restrict__ means,
    const float* __restrict__ inv_stds,
    const float4* __restrict__ weight,
    const float4* __restrict__ bias,
    float4* __restrict__ output,
    int64_t N_over_4) {
    
    int group_idx = blockIdx.y;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N_over_4) {
        float mean = means[group_idx];
        float inv_std = inv_stds[group_idx];
        
        int64_t idx = (int64_t)group_idx * N_over_4 + i;
        float4 in4 = input[idx];
        float4 w4 = weight[i];
        float4 b4 = bias[i];
        
        float4 out4;
        out4.x = (in4.x - mean) * inv_std * w4.x + b4.x;
        out4.y = (in4.y - mean) * inv_std * w4.y + b4.y;
        out4.z = (in4.z - mean) * inv_std * w4.z + b4.z;
        out4.w = (in4.w - mean) * inv_std * w4.w + b4.w;
        
        output[idx] = out4;
    }
}

__global__ void layernorm_forward_kernel3_basic(
    const float* __restrict__ input,
    const float* __restrict__ means,
    const float* __restrict__ inv_stds,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t N) {
    
    int group_idx = blockIdx.y;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        float mean = means[group_idx];
        float inv_std = inv_stds[group_idx];
        float w = (weight != nullptr) ? weight[i] : 1.0f;
        float b = (bias != nullptr) ? bias[i] : 0.0f;
        
        int64_t idx = (int64_t)group_idx * N + i;
        output[idx] = (input[idx] - mean) * inv_std * w + b;
    }
}

torch::Tensor layernorm_hip(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, std::vector<int64_t> normalized_shape, float eps) {
    int64_t N = 1;
    for (auto s : normalized_shape) {
        N *= s;
    }
    int64_t M = x.numel() / N;
    
    auto output = torch::empty_like(x);
    auto means = torch::empty({M}, x.options());
    auto inv_stds = torch::empty({M}, x.options());
    
    int num_blocks_per_group = 512;
    auto partial_sums = torch::empty({M * num_blocks_per_group}, x.options().dtype(torch::kFloat64));
    auto partial_sq_sums = torch::empty({M * num_blocks_per_group}, x.options().dtype(torch::kFloat64));
    
    if (N % 4 == 0) {
        int64_t N_over_4 = N / 4;
        dim3 block1(1024);
        dim3 grid1(num_blocks_per_group, M);
        layernorm_forward_kernel1_float4<<<grid1, block1>>>(
            (const float4*)x.data_ptr<float>(),
            partial_sums.data_ptr<double>(),
            partial_sq_sums.data_ptr<double>(),
            N_over_4,
            num_blocks_per_group
        );
        
        dim3 block2(1024);
        dim3 grid2(M);
        layernorm_forward_kernel2<<<grid2, block2>>>(
            partial_sums.data_ptr<double>(),
            partial_sq_sums.data_ptr<double>(),
            means.data_ptr<float>(),
            inv_stds.data_ptr<float>(),
            N,
            num_blocks_per_group,
            eps
        );
        
        dim3 block3(1024);
        dim3 grid3((N_over_4 + block3.x - 1) / block3.x, M);
        layernorm_forward_kernel3_float4<<<grid3, block3>>>(
            (const float4*)x.data_ptr<float>(),
            means.data_ptr<float>(),
            inv_stds.data_ptr<float>(),
            (const float4*)weight.data_ptr<float>(),
            (const float4*)bias.data_ptr<float>(),
            (float4*)output.data_ptr<float>(),
            N_over_4
        );
    } else {
        dim3 block1(1024);
        dim3 grid1(num_blocks_per_group, M);
        layernorm_forward_kernel1_basic<<<grid1, block1>>>(
            x.data_ptr<float>(),
            partial_sums.data_ptr<double>(),
            partial_sq_sums.data_ptr<double>(),
            N,
            num_blocks_per_group
        );
        
        dim3 block2(1024);
        dim3 grid2(M);
        layernorm_forward_kernel2<<<grid2, block2>>>(
            partial_sums.data_ptr<double>(),
            partial_sq_sums.data_ptr<double>(),
            means.data_ptr<float>(),
            inv_stds.data_ptr<float>(),
            N,
            num_blocks_per_group,
            eps
        );
        
        dim3 block3(1024);
        dim3 grid3((N + block3.x - 1) / block3.x, M);
        layernorm_forward_kernel3_basic<<<grid3, block3>>>(
            x.data_ptr<float>(),
            means.data_ptr<float>(),
            inv_stds.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            output.data_ptr<float>(),
            N
        );
    }
    
    return output;
}
"""

layernorm_module = load_inline(
    name="layernorm_hip",
    cpp_sources=layernorm_hip_source,
    functions=["layernorm_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return layernorm_module.layernorm_hip(x, self.ln.weight, self.ln.bias, self.ln.normalized_shape, self.ln.eps)

