
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

layernorm_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define WARP_SIZE 64

__device__ inline void warp_reduce_sum_double(float &sum, float &sq_sum) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset);
        sq_sum += __shfl_down(sq_sum, offset);
    }
}

__device__ inline void block_reduce_sum_double(float &sum, float &sq_sum, float* shared_sum, float* shared_sq_sum) {
    int tid = threadIdx.x;
    int wid = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    warp_reduce_sum_double(sum, sq_sum);

    if (lane == 0) {
        shared_sum[wid] = sum;
        shared_sq_sum[wid] = sq_sum;
    }
    __syncthreads();

    if (wid == 0) {
        sum = (tid < (blockDim.x / WARP_SIZE)) ? shared_sum[tid] : 0.0f;
        sq_sum = (tid < (blockDim.x / WARP_SIZE)) ? shared_sq_sum[tid] : 0.0f;
        warp_reduce_sum_double(sum, sq_sum);
        if (tid == 0) {
            shared_sum[0] = sum;
            shared_sq_sum[0] = sq_sum;
        }
    }
}

__global__ void layernorm_kernel(const float* __restrict__ x, 
                                const float* __restrict__ weight, 
                                const float* __restrict__ bias, 
                                float* __restrict__ out, 
                                int M, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_dim = blockDim.x;

    __shared__ float shared_sum_mem[32]; 
    __shared__ float shared_sq_sum_mem[32];

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    const float4* x4 = reinterpret_cast<const float4*>(x + row * M);
    int M4 = M / 4;

    for (int i = tid; i < M4; i += block_dim) {
        float4 val4 = x4[i];
        local_sum += val4.x + val4.y + val4.z + val4.w;
        local_sq_sum += val4.x * val4.x + val4.y * val4.y + val4.z * val4.z + val4.w * val4.w;
    }

    if (M4 * 4 + tid < M) {
        for (int i = M4 * 4 + tid; i < M; i += block_dim) {
            float val = x[row * M + i];
            local_sum += val;
            local_sq_sum += val * val;
        }
    }

    block_reduce_sum_double(local_sum, local_sq_sum, shared_sum_mem, shared_sq_sum_mem);
    __syncthreads();

    float mean = shared_sum_mem[0] / M;
    float var = fmaxf(0.0f, (shared_sq_sum_mem[0] / M) - (mean * mean));
    float inv_std = 1.0f / sqrtf(var + eps);

    float4* out4 = reinterpret_cast<float4*>(out + row * M);
    const float4* w4 = reinterpret_cast<const float4*>(weight);
    const float4* b4 = reinterpret_cast<const float4*>(bias);

    for (int i = tid; i < M4; i += block_dim) {
        float4 val4 = x4[i];
        float4 weight4 = w4[i];
        float4 bias4 = b4[i];
        
        float4 res;
        res.x = (val4.x - mean) * inv_std * weight4.x + bias4.x;
        res.y = (val4.y - mean) * inv_std * weight4.y + bias4.y;
        res.z = (val4.z - mean) * inv_std * weight4.z + bias4.z;
        res.w = (val4.w - mean) * inv_std * weight4.w + bias4.w;
        out4[i] = res;
    }

    if (M4 * 4 + tid < M) {
        for (int i = M4 * 4 + tid; i < M; i += block_dim) {
            out[row * M + i] = (x[row * M + i] - mean) * inv_std * weight[i] + bias[i];
        }
    }
}

torch::Tensor layernorm_hip(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float eps) {
    auto input_shape = x.sizes();
    int N = input_shape[0];
    int M = 1;
    for (int i = 1; i < input_shape.size(); ++i) {
        M *= input_shape[i];
    }

    auto out = torch::empty_like(x);

    const int block_size = 1024;
    layernorm_kernel<<<N, block_size>>>(
        x.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        out.data_ptr<float>(), 
        M, eps
    );

    return out;
}
"""

layernorm_cpp_source = """
torch::Tensor layernorm_hip(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float eps);
"""

layernorm_lib = load_inline(
    name="layernorm_lib",
    cpp_sources=layernorm_cpp_source,
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
        original_shape = x.shape
        M = 1
        for s in self.normalized_shape:
            M *= s
        N = x.numel() // M
        
        x_reshaped = x.view(N, M)
        weight_flat = self.weight.view(-1)
        bias_flat = self.bias.view(-1)
        
        # Ensure weight and bias are contiguous
        weight_flat = weight_flat.contiguous()
        bias_flat = bias_flat.contiguous()
        x_reshaped = x_reshaped.contiguous()
        
        out = layernorm_lib.layernorm_hip(x_reshaped, weight_flat, bias_flat, self.eps)
        return out.view(original_shape)

