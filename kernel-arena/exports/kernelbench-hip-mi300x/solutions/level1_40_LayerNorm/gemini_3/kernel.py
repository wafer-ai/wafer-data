import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Helper for reduction
__device__ __forceinline__ void warpReduceSum(float& val) {
    val += __shfl_down(val, 16);
    val += __shfl_down(val, 8);
    val += __shfl_down(val, 4);
    val += __shfl_down(val, 2);
    val += __shfl_down(val, 1);
}

__device__ __forceinline__ void blockReduceSum(float& val) {
    static __shared__ float shared[32]; // Shared mem for warp sums
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    warpReduceSum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads(); 

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) warpReduceSum(val);
}

__device__ __forceinline__ void warpReduceSum2(float& v1, float& v2) {
    v1 += __shfl_down(v1, 16); v2 += __shfl_down(v2, 16);
    v1 += __shfl_down(v1, 8);  v2 += __shfl_down(v2, 8);
    v1 += __shfl_down(v1, 4);  v2 += __shfl_down(v2, 4);
    v1 += __shfl_down(v1, 2);  v2 += __shfl_down(v2, 2);
    v1 += __shfl_down(v1, 1);  v2 += __shfl_down(v2, 1);
}

// Reduce two values: sum and sum_sq
__device__ __forceinline__ void blockReduceSum2(float& sum, float& sq_sum) {
    static __shared__ float shared_sum[32];
    static __shared__ float shared_sq[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    warpReduceSum2(sum, sq_sum);

    if (lane == 0) {
        shared_sum[wid] = sum;
        shared_sq[wid] = sq_sum;
    }
    __syncthreads();

    if (wid == 0) {
        sum = (threadIdx.x < blockDim.x / 32) ? shared_sum[lane] : 0.0f;
        sq_sum = (threadIdx.x < blockDim.x / 32) ? shared_sq[lane] : 0.0f;
        warpReduceSum2(sum, sq_sum);
    }
}

// Kernel 1: Partial Reduction using float4
// Grid: (num_splits, batch_size)
// Block: 256
__global__ void part_reduce_kernel(const float* __restrict__ x, float* __restrict__ partials, int N_vec, int num_splits) {
    int batch_idx = blockIdx.y;
    int split_idx = blockIdx.x;
    int tid = threadIdx.x;

    // Calculate the range of vectors this block is responsible for
    int vectors_per_split = (N_vec + num_splits - 1) / num_splits;
    int start_vec = split_idx * vectors_per_split;
    int end_vec = min(start_vec + vectors_per_split, N_vec);
    
    // Pointer to the start of this batch's data
    // x is contiguous (Batch, N) -> (Batch, N_vec * 4)
    const float4* x_vec = (const float4*)x + (batch_idx * N_vec);

    float sum = 0.0f;
    float sq_sum = 0.0f;

    for (int i = start_vec + tid; i < end_vec; i += blockDim.x) {
        float4 val = x_vec[i];
        sum += val.x + val.y + val.z + val.w;
        sq_sum += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }

    blockReduceSum2(sum, sq_sum);

    if (tid == 0) {
        int out_idx = (batch_idx * num_splits + split_idx) * 2;
        partials[out_idx] = sum;
        partials[out_idx + 1] = sq_sum;
    }
}

// Kernel 2: Final Reduction
// Grid: (batch_size, 1)
// Block: 256 (should be >= num_splits if possible, or loop)
// Here we assume num_splits <= 1024. If block size is 256, we loop.
__global__ void final_reduce_kernel(const float* __restrict__ partials, float* __restrict__ stats, int num_splits, int N, float eps) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    float sum = 0.0f;
    float sq_sum = 0.0f;

    // Base offset for this batch in partials array
    const float* batch_partials = partials + (batch_idx * num_splits * 2);

    for (int i = tid; i < num_splits; i += blockDim.x) {
        sum += batch_partials[i * 2];
        sq_sum += batch_partials[i * 2 + 1];
    }

    blockReduceSum2(sum, sq_sum);

    if (tid == 0) {
        float mean = sum / N;
        float var = (sq_sum / N) - (mean * mean);
        if (var < 0.0f) var = 0.0f;
        float rstd = rsqrtf(var + eps);
        stats[batch_idx * 2] = mean;
        stats[batch_idx * 2 + 1] = rstd;
    }
}

// Kernel 3: Apply Normalization and Affine Transform
// Grid: (N_vec / 256, batch_size) (approx)
// Block: 256
__global__ void apply_ln_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ stats,
    float* __restrict__ y,
    int N_vec) 
{
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;
    int idx_vec = blockIdx.x * blockDim.x + tid;

    if (idx_vec < N_vec) {
        // Load stats
        float mean = stats[batch_idx * 2];
        float rstd = stats[batch_idx * 2 + 1];

        // Load data
        const float4* x_ptr = (const float4*)x + (batch_idx * N_vec);
        const float4* g_ptr = (const float4*)gamma;
        const float4* b_ptr = (const float4*)beta;
        float4* y_ptr = (float4*)y + (batch_idx * N_vec);

        float4 val = x_ptr[idx_vec];
        float4 g = g_ptr[idx_vec];
        float4 b = b_ptr[idx_vec];
        float4 out;

        out.x = (val.x - mean) * rstd * g.x + b.x;
        out.y = (val.y - mean) * rstd * g.y + b.y;
        out.z = (val.z - mean) * rstd * g.z + b.z;
        out.w = (val.w - mean) * rstd * g.w + b.w;

        y_ptr[idx_vec] = out;
    }
}

torch::Tensor layer_norm_hip(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float eps) {
    // x: (Batch, Features, Dim1, Dim2) -> flattened to (Batch, N)
    // gamma, beta: (Features, Dim1, Dim2) -> flattened to (N)
    
    // Ensure inputs are contiguous
    x = x.contiguous();
    gamma = gamma.contiguous();
    beta = beta.contiguous();
    
    int batch_size = x.size(0);
    int N = x.numel() / batch_size;
    
    // Check for float4 alignment/divisibility
    if (N % 4 != 0) {
        // Fallback or error. For this problem, N = 64*256*256 is div by 4.
        // If not, we would need a scalar kernel.
        return torch::layer_norm(x, gamma.sizes(), gamma, beta, eps);
    }
    
    int N_vec = N / 4;
    auto y = torch::empty_like(x);
    
    // Allocate temp buffers
    // num_splits = 1024 seems good for N=4M (4096 elements per split)
    int num_splits = 1024;
    auto partials = torch::empty({batch_size, num_splits, 2}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    auto stats = torch::empty({batch_size, 2}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    
    // 1. Partial Reduce
    dim3 block_reduce(256);
    dim3 grid_reduce(num_splits, batch_size);
    part_reduce_kernel<<<grid_reduce, block_reduce>>>(x.data_ptr<float>(), partials.data_ptr<float>(), N_vec, num_splits);
    
    // 2. Final Reduce
    dim3 block_final(256);
    dim3 grid_final(batch_size, 1);
    final_reduce_kernel<<<grid_final, block_final>>>(partials.data_ptr<float>(), stats.data_ptr<float>(), num_splits, N, eps);
    
    // 3. Apply
    dim3 block_apply(256);
    int grid_x = (N_vec + 255) / 256;
    dim3 grid_apply(grid_x, batch_size);
    apply_ln_kernel<<<grid_apply, block_apply>>>(
        x.data_ptr<float>(), 
        gamma.data_ptr<float>(), 
        beta.data_ptr<float>(), 
        stats.data_ptr<float>(), 
        y.data_ptr<float>(), 
        N_vec
    );
    
    return y;
}
"""

ln_module = load_inline(
    name="layer_norm_kernels",
    cpp_sources=cpp_source,
    functions=["layer_norm_hip"],
    verbose=True,
    extra_cflags=['-O3']
)

class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = 1e-5
        # Initialize parameters to match nn.LayerNorm defaults
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ln_module.layer_norm_hip(x, self.weight, self.bias, self.eps)

