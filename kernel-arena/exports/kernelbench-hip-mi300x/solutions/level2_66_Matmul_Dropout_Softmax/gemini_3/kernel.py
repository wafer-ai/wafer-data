
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>
#include <cmath>

// float4 is built-in

__device__ __forceinline__ float rand_uniform(unsigned int idx, unsigned int seed) {
    unsigned int h = idx ^ seed;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return (float)h / 4294967296.0f;
}

__device__ __forceinline__ float warpReduceMax(float val) {
    int ws = warpSize; 
    for (int offset = ws / 2; offset > 0; offset /= 2)
        val = max(val, __shfl_down(val, offset));
    return val;
}

__device__ __forceinline__ float warpReduceSum(float val) {
    int ws = warpSize;
    for (int offset = ws / 2; offset > 0; offset /= 2)
        val += __shfl_down(val, offset);
    return val;
}

__global__ void reduce_step1_kernel(
    const float* __restrict__ input,
    float* __restrict__ workspace_max,
    float* __restrict__ workspace_sum,
    int rows,
    int cols,
    int chunks_per_row,
    float p,
    float scale,
    unsigned int seed
) {
    int row = blockIdx.x;
    int chunk = blockIdx.y;
    int tid = threadIdx.x;
    int ws = warpSize;
    int lane = tid % ws;
    int warp_id = tid / ws;
    int num_warps = blockDim.x / ws;
    
    int chunk_size = cols / chunks_per_row;
    int start_col = chunk * chunk_size;
    const float* row_in = input + row * cols;
    const float4* row_in_vec = reinterpret_cast<const float4*>(row_in);
    
    float local_max = -1e30f;
    float local_sum = 0.0f;
    
    int vec_start = start_col / 4;
    int vec_end = vec_start + (chunk_size / 4);
    
    for (int i = vec_start + tid; i < vec_end; i += blockDim.x) {
        float4 v = row_in_vec[i];
        float vals[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float val = vals[k];
            if (p > 0.0f) {
                unsigned int global_idx = row * cols + (i * 4 + k);
                float r = rand_uniform(global_idx, seed);
                if (r < p) val = 0.0f; else val *= scale;
            }
            if (val > local_max) {
                float diff = local_max - val;
                local_sum = local_sum * expf(diff) + 1.0f;
                local_max = val;
            } else {
                local_sum += expf(val - local_max);
            }
        }
    }
    
    __shared__ float s_max[32];
    __shared__ float s_sum[32];
    
    float warp_max = warpReduceMax(local_max);
    if (lane == 0) s_max[warp_id] = warp_max;
    __syncthreads();
    
    if (tid == 0) {
        float m = -1e30f;
        for (int i=0; i<num_warps; ++i) m = max(m, s_max[i]);
        s_max[0] = m;
    }
    __syncthreads();
    float block_max = s_max[0];
    
    local_sum = local_sum * expf(local_max - block_max);
    float warp_sum = warpReduceSum(local_sum);
    if (lane == 0) s_sum[warp_id] = warp_sum;
    __syncthreads();
    
    if (tid == 0) {
        float s = 0.0f;
        for (int i=0; i<num_warps; ++i) s += s_sum[i];
        workspace_max[row * chunks_per_row + chunk] = block_max;
        workspace_sum[row * chunks_per_row + chunk] = s;
    }
}

__global__ void reduce_step2_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ workspace_max,
    const float* __restrict__ workspace_sum,
    int rows,
    int cols,
    int chunks_per_row,
    float p,
    float scale,
    unsigned int seed
) {
    int row = blockIdx.x;
    int chunk = blockIdx.y;
    int tid = threadIdx.x;
    
    __shared__ float s_global_max;
    __shared__ float s_global_sum;
    
    if (tid == 0) {
        float g_max = -1e30f;
        for (int c = 0; c < chunks_per_row; ++c) {
            float m = workspace_max[row * chunks_per_row + c];
            if (m > g_max) g_max = m;
        }
        s_global_max = g_max;
        
        float g_sum = 0.0f;
        for (int c = 0; c < chunks_per_row; ++c) {
            float m = workspace_max[row * chunks_per_row + c];
            float s = workspace_sum[row * chunks_per_row + c];
            g_sum += s * expf(m - g_max);
        }
        s_global_sum = g_sum;
    }
    __syncthreads();
    
    float global_max = s_global_max;
    float global_sum = s_global_sum;
    
    int chunk_size = cols / chunks_per_row;
    int start_col = chunk * chunk_size;
    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;
    
    const float4* row_in_vec = reinterpret_cast<const float4*>(row_in);
    float4* row_out_vec = reinterpret_cast<float4*>(row_out);
    
    int vec_start = start_col / 4;
    int vec_end = vec_start + (chunk_size / 4);
    
    for (int i = vec_start + tid; i < vec_end; i += blockDim.x) {
        float4 v = row_in_vec[i];
        float vals[4] = {v.x, v.y, v.z, v.w};
        float4 out_v;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float val = vals[k];
            if (p > 0.0f) {
                unsigned int global_idx = row * cols + (i * 4 + k);
                float r = rand_uniform(global_idx, seed);
                if (r < p) val = 0.0f; else val *= scale;
            }
            vals[k] = expf(val - global_max) / global_sum;
        }
        out_v.x = vals[0]; out_v.y = vals[1]; out_v.z = vals[2]; out_v.w = vals[3];
        row_out_vec[i] = out_v;
    }
}

torch::Tensor fused_dropout_softmax_split(torch::Tensor input, double p, double scale, int64_t seed, int64_t stream_ptr) {
    auto rows = input.size(0);
    auto cols = input.size(1);
    auto output = torch::empty_like(input);
    int chunks_per_row = 16;
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    auto workspace_max = torch::empty({rows, chunks_per_row}, options);
    auto workspace_sum = torch::empty({rows, chunks_per_row}, options);
    
    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);
    
    dim3 grid(rows, chunks_per_row);
    int block_size = 256;
    
    reduce_step1_kernel<<<grid, block_size, 0, stream>>>(
        input.data_ptr<float>(),
        workspace_max.data_ptr<float>(),
        workspace_sum.data_ptr<float>(),
        rows, cols, chunks_per_row,
        (float)p, (float)scale, (unsigned int)seed
    );
    
    reduce_step2_kernel<<<grid, block_size, 0, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        workspace_max.data_ptr<float>(),
        workspace_sum.data_ptr<float>(),
        rows, cols, chunks_per_row,
        (float)p, (float)scale, (unsigned int)seed
    );
    
    return output;
}
"""

fused_ops = load_inline(
    name="fused_dropout_softmax_v5",
    cpp_sources=cpp_source,
    functions=["fused_dropout_softmax_split"],
    extra_include_paths=[],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout_p = dropout_p
        self.fused_op = fused_ops

    def forward(self, x):
        x = self.matmul(x)
        if self.training and self.dropout_p > 0.0:
            p = self.dropout_p
            scale = 1.0 / (1.0 - p)
            seed = torch.cuda.initial_seed() & 0xFFFFFFFF
        else:
            p = 0.0
            scale = 1.0
            seed = 0
        stream = torch.cuda.current_stream().cuda_stream
        return self.fused_op.fused_dropout_softmax_split(x, p, scale, seed, stream)
