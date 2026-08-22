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

// Optimized multi-phase LayerNorm with better occupancy

// Phase 1: Compute partial sums per block using Welford's algorithm
// Uses atomic add to directly accumulate into batch-level sums
__global__ void layernorm_compute_partial_v2(
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
    
    const float* x = input + (size_t)batch_idx * norm_size;
    
    // Each block handles a portion of the normalized dimension
    const int elements_per_block = (norm_size + blocks_per_batch - 1) / blocks_per_batch;
    const int start = block_in_batch * elements_per_block;
    const int end = min(start + elements_per_block, norm_size);
    
    float local_sum = 0.0f;
    float local_sqsum = 0.0f;
    
    // Vectorized loads - 4 floats at a time
    // Ensure alignment
    const int vec_start = (start + 3) / 4;  // First aligned vector
    const int vec_end = end / 4;
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    
    // Pre-vector scalar elements
    for (int i = start + threadIdx.x; i < min(vec_start * 4, end); i += blockDim.x) {
        float val = x[i];
        local_sum += val;
        local_sqsum += val * val;
    }
    
    // Vectorized elements
    for (int i = vec_start + threadIdx.x; i < vec_end; i += blockDim.x) {
        float4 val = x_vec[i];
        local_sum += val.x + val.y + val.z + val.w;
        local_sqsum += val.x*val.x + val.y*val.y + val.z*val.z + val.w*val.w;
    }
    
    // Post-vector scalar elements
    for (int i = vec_end * 4 + threadIdx.x; i < end; i += blockDim.x) {
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
    
    if (threadIdx.x == 0) {
        partial_sum[blockIdx.x] = local_sum;
        partial_sqsum[blockIdx.x] = local_sqsum;
    }
}

// Phase 2: Reduce partial sums and compute mean/inv_std
__global__ void layernorm_reduce_stats_v2(
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
    
    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down(local_sum, offset, WARP_SIZE);
        local_sqsum += __shfl_down(local_sqsum, offset, WARP_SIZE);
    }
    
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid = threadIdx.x / WARP_SIZE;
    const int numWarps = blockDim.x / WARP_SIZE;
    
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
    
    if (threadIdx.x == 0) {
        float mean = local_sum / norm_size;
        float var = local_sqsum / norm_size - mean * mean;
        mean_out[batch_idx] = mean;
        inv_std_out[batch_idx] = rsqrtf(var + eps);
    }
}

// Phase 3: Apply normalization with vectorized loads/stores
__global__ void layernorm_apply_v2(
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
    
    const float* x = input + (size_t)batch_idx * norm_size;
    float* y = output + (size_t)batch_idx * norm_size;
    
    const float m = mean[batch_idx];
    const float istd = inv_std[batch_idx];
    
    const int num_vec = norm_size / 4;
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (global_idx < num_vec) {
        const float4* x_vec = reinterpret_cast<const float4*>(x);
        float4* y_vec = reinterpret_cast<float4*>(y);
        const float4* g_vec = reinterpret_cast<const float4*>(gamma);
        const float4* b_vec = reinterpret_cast<const float4*>(beta);
        
        float4 val = x_vec[global_idx];
        float4 g = g_vec[global_idx];
        float4 b = b_vec[global_idx];
        
        float4 out;
        out.x = (val.x - m) * istd * g.x + b.x;
        out.y = (val.y - m) * istd * g.y + b.y;
        out.z = (val.z - m) * istd * g.z + b.z;
        out.w = (val.w - m) * istd * g.w + b.w;
        
        y_vec[global_idx] = out;
    }
    
    // Handle remainder
    const int scalar_idx = num_vec * 4 + (blockIdx.x * blockDim.x + threadIdx.x - num_vec);
    if (scalar_idx >= num_vec * 4 && scalar_idx < norm_size && global_idx >= num_vec) {
        float val = x[scalar_idx];
        y[scalar_idx] = (val - m) * istd * gamma[scalar_idx] + beta[scalar_idx];
    }
}

torch::Tensor layernorm_hip(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, double eps) {
    auto batch_size = input.size(0);
    auto norm_size = input.numel() / batch_size;
    
    auto output = torch::empty_like(input);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    
    // Use multiple blocks per batch for reduction
    // More blocks = more parallelism but more reduction overhead
    const int blocks_per_batch = 128;
    
    // Allocate partial sums
    auto partial_sum = torch::empty({batch_size * blocks_per_batch}, options);
    auto partial_sqsum = torch::empty({batch_size * blocks_per_batch}, options);
    auto mean_tensor = torch::empty({batch_size}, options);
    auto inv_std_tensor = torch::empty({batch_size}, options);
    
    // Phase 1: Compute partial sums
    const int block_size_1 = 512;
    const int num_blocks_1 = batch_size * blocks_per_batch;
    layernorm_compute_partial_v2<<<num_blocks_1, block_size_1>>>(
        input.data_ptr<float>(),
        partial_sum.data_ptr<float>(),
        partial_sqsum.data_ptr<float>(),
        batch_size,
        norm_size,
        blocks_per_batch
    );
    
    // Phase 2: Reduce and compute stats
    const int block_size_2 = 256;
    layernorm_reduce_stats_v2<<<batch_size, block_size_2>>>(
        partial_sum.data_ptr<float>(),
        partial_sqsum.data_ptr<float>(),
        mean_tensor.data_ptr<float>(),
        inv_std_tensor.data_ptr<float>(),
        batch_size,
        norm_size,
        blocks_per_batch,
        (float)eps
    );
    
    // Phase 3: Apply normalization
    const int block_size_3 = 256;
    const int norm_blocks = (norm_size / 4 + block_size_3 - 1) / block_size_3;
    dim3 grid_3(norm_blocks, batch_size);
    layernorm_apply_v2<<<grid_3, block_size_3>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        mean_tensor.data_ptr<float>(),
        inv_std_tensor.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        norm_size
    );
    
    return output;
}
"""

layernorm_module = load_inline(
    name="layernorm_hip_v3",
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
