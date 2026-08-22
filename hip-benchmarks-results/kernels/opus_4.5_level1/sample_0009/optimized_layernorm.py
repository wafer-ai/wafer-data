import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

layernorm_cpp_source = """
torch::Tensor layernorm_hip(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, double eps);
"""

layernorm_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

#define WARP_SIZE 64

// Welford's online algorithm for computing mean and variance in one pass
// Uses two-phase reduction: first compute partial sums per block, then combine

template<int BLOCK_SIZE>
__device__ __forceinline__ void warpReduceSumPair(float& sum, float& sqsum) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, WARP_SIZE);
        sqsum += __shfl_down(sqsum, offset, WARP_SIZE);
    }
}

template<int BLOCK_SIZE>
__device__ void blockReduceSumPair(float& sum, float& sqsum, float* shared_sum, float* shared_sqsum) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid = threadIdx.x / WARP_SIZE;
    const int numWarps = BLOCK_SIZE / WARP_SIZE;
    
    warpReduceSumPair<BLOCK_SIZE>(sum, sqsum);
    
    if (lane == 0) {
        shared_sum[wid] = sum;
        shared_sqsum[wid] = sqsum;
    }
    __syncthreads();
    
    sum = (threadIdx.x < numWarps) ? shared_sum[threadIdx.x] : 0.0f;
    sqsum = (threadIdx.x < numWarps) ? shared_sqsum[threadIdx.x] : 0.0f;
    
    if (wid == 0) {
        warpReduceSumPair<BLOCK_SIZE>(sum, sqsum);
    }
}

// First kernel: compute partial sums per block
__global__ void layernorm_partial_sums(
    const float* __restrict__ input,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_sqsum,
    int batch_size,
    int norm_size,
    int blocks_per_batch
) {
    __shared__ float shared_sum[16];
    __shared__ float shared_sqsum[16];
    
    const int batch_idx = blockIdx.x / blocks_per_batch;
    const int block_in_batch = blockIdx.x % blocks_per_batch;
    
    if (batch_idx >= batch_size) return;
    
    const float* x = input + batch_idx * norm_size;
    
    const int elements_per_block = (norm_size + blocks_per_batch - 1) / blocks_per_batch;
    const int start_idx = block_in_batch * elements_per_block;
    const int end_idx = min(start_idx + elements_per_block, norm_size);
    
    float local_sum = 0.0f;
    float local_sqsum = 0.0f;
    
    for (int i = start_idx + threadIdx.x; i < end_idx; i += blockDim.x) {
        float val = x[i];
        local_sum += val;
        local_sqsum += val * val;
    }
    
    blockReduceSumPair<1024>(local_sum, local_sqsum, shared_sum, shared_sqsum);
    
    if (threadIdx.x == 0) {
        partial_sum[blockIdx.x] = local_sum;
        partial_sqsum[blockIdx.x] = local_sqsum;
    }
}

// Second kernel: reduce partial sums and compute final statistics
__global__ void layernorm_reduce_stats(
    const float* __restrict__ partial_sum,
    const float* __restrict__ partial_sqsum,
    float* __restrict__ mean_out,
    float* __restrict__ inv_std_out,
    int batch_size,
    int norm_size,
    int blocks_per_batch,
    float eps
) {
    __shared__ float shared_sum[16];
    __shared__ float shared_sqsum[16];
    
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* ps = partial_sum + batch_idx * blocks_per_batch;
    const float* psq = partial_sqsum + batch_idx * blocks_per_batch;
    
    float local_sum = 0.0f;
    float local_sqsum = 0.0f;
    
    for (int i = threadIdx.x; i < blocks_per_batch; i += blockDim.x) {
        local_sum += ps[i];
        local_sqsum += psq[i];
    }
    
    blockReduceSumPair<256>(local_sum, local_sqsum, shared_sum, shared_sqsum);
    
    if (threadIdx.x == 0) {
        float mean = local_sum / norm_size;
        float var = local_sqsum / norm_size - mean * mean;
        mean_out[batch_idx] = mean;
        inv_std_out[batch_idx] = rsqrtf(var + eps);
    }
}

// Third kernel: apply normalization
__global__ void layernorm_normalize(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean,
    const float* __restrict__ inv_std,
    float* __restrict__ output,
    int batch_size,
    int norm_size
) {
    const int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= norm_size) return;
    
    const float* x = input + batch_idx * norm_size;
    float* y = output + batch_idx * norm_size;
    
    float m = mean[batch_idx];
    float istd = inv_std[batch_idx];
    
    float normalized = (x[idx] - m) * istd;
    y[idx] = normalized * gamma[idx] + beta[idx];
}

// Fused single-kernel approach for better performance
__global__ void layernorm_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int batch_size,
    int norm_size,
    float eps
) {
    extern __shared__ float shared[];
    
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* x = input + (size_t)batch_idx * norm_size;
    float* y = output + (size_t)batch_idx * norm_size;
    
    // Two-pass algorithm: first compute mean, then variance
    // Using vectorized loads for better memory bandwidth
    
    float local_sum = 0.0f;
    float local_sqsum = 0.0f;
    
    // Process 4 elements at a time when possible
    const int vec_size = 4;
    const int num_vec = norm_size / vec_size;
    const int remainder = norm_size % vec_size;
    
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    
    for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
        float4 val = x_vec[i];
        local_sum += val.x + val.y + val.z + val.w;
        local_sqsum += val.x*val.x + val.y*val.y + val.z*val.z + val.w*val.w;
    }
    
    // Handle remainder
    for (int i = num_vec * vec_size + threadIdx.x; i < norm_size; i += blockDim.x) {
        float val = x[i];
        local_sum += val;
        local_sqsum += val * val;
    }
    
    // Block reduction
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid = threadIdx.x / WARP_SIZE;
    const int numWarps = blockDim.x / WARP_SIZE;
    
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down(local_sum, offset, WARP_SIZE);
        local_sqsum += __shfl_down(local_sqsum, offset, WARP_SIZE);
    }
    
    float* shared_sum = shared;
    float* shared_sqsum = shared + numWarps;
    
    if (lane == 0) {
        shared_sum[wid] = local_sum;
        shared_sqsum[wid] = local_sqsum;
    }
    __syncthreads();
    
    if (threadIdx.x < numWarps) {
        local_sum = shared_sum[threadIdx.x];
        local_sqsum = shared_sqsum[threadIdx.x];
    } else {
        local_sum = 0.0f;
        local_sqsum = 0.0f;
    }
    
    if (wid == 0) {
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            local_sum += __shfl_down(local_sum, offset, WARP_SIZE);
            local_sqsum += __shfl_down(local_sqsum, offset, WARP_SIZE);
        }
    }
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        float mean = local_sum / norm_size;
        float var = local_sqsum / norm_size - mean * mean;
        shared[0] = mean;
        shared[1] = rsqrtf(var + eps);
    }
    __syncthreads();
    
    float mean = shared[0];
    float inv_std = shared[1];
    
    // Apply normalization with vectorized writes
    float4* y_vec = reinterpret_cast<float4*>(y);
    const float4* gamma_vec = reinterpret_cast<const float4*>(gamma);
    const float4* beta_vec = reinterpret_cast<const float4*>(beta);
    
    for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
        float4 val = x_vec[i];
        float4 g = gamma_vec[i];
        float4 b = beta_vec[i];
        
        float4 out;
        out.x = (val.x - mean) * inv_std * g.x + b.x;
        out.y = (val.y - mean) * inv_std * g.y + b.y;
        out.z = (val.z - mean) * inv_std * g.z + b.z;
        out.w = (val.w - mean) * inv_std * g.w + b.w;
        
        y_vec[i] = out;
    }
    
    // Handle remainder
    for (int i = num_vec * vec_size + threadIdx.x; i < norm_size; i += blockDim.x) {
        float val = x[i];
        float normalized = (val - mean) * inv_std;
        y[i] = normalized * gamma[i] + beta[i];
    }
}

torch::Tensor layernorm_hip(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, double eps) {
    auto batch_size = input.size(0);
    auto norm_size = input.numel() / batch_size;
    
    auto output = torch::empty_like(input);
    
    // Use fused kernel with maximum threads
    const int block_size = 1024;
    const int num_blocks = batch_size;
    const int num_warps = block_size / WARP_SIZE;
    const int shared_mem_size = (2 * num_warps + 2) * sizeof(float);
    
    layernorm_fused_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        norm_size,
        (float)eps
    );
    
    return output;
}
"""

layernorm_module = load_inline(
    name="layernorm_hip",
    cpp_sources=layernorm_cpp_source,
    cuda_sources=layernorm_hip_source,
    functions=["layernorm_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = 1e-5
        
        # Initialize gamma (weight) and beta (bias) like nn.LayerNorm
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return layernorm_module.layernorm_hip(
            x.contiguous(),
            self.gamma.contiguous().view(-1),
            self.beta.contiguous().view(-1),
            self.eps
        )


batch_size = 16
features = 64
dim1 = 256
dim2 = 256


def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]


def get_init_inputs():
    return [(features, dim1, dim2)]
