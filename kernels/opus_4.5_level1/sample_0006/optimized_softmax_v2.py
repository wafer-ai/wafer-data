import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

softmax_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <float.h>

// Warp size on AMD GPUs
#define WARP_SIZE 64

// Warp reduction for max
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

// Warp reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Vectorized load using float4
__device__ __forceinline__ float4 load_float4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

// Vectorized store using float4
__device__ __forceinline__ void store_float4(float* ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
}

// Optimized softmax kernel with vectorized memory access
__global__ void softmax_kernel_vec4(const float* __restrict__ input, 
                                     float* __restrict__ output,
                                     int num_rows, int num_cols) {
    extern __shared__ float smem[];
    
    int row = blockIdx.x;
    if (row >= num_rows) return;
    
    const float* row_in = input + row * num_cols;
    float* row_out = output + row * num_cols;
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int num_warps = num_threads / WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // Number of float4s (process 4 elements at a time)
    int vec_cols = num_cols / 4;
    int rem_start = vec_cols * 4;
    
    // Phase 1: Find maximum using vectorized loads
    float local_max = -FLT_MAX;
    
    // Process float4 chunks
    for (int i = tid; i < vec_cols; i += num_threads) {
        float4 v = load_float4(row_in + i * 4);
        local_max = fmaxf(local_max, v.x);
        local_max = fmaxf(local_max, v.y);
        local_max = fmaxf(local_max, v.z);
        local_max = fmaxf(local_max, v.w);
    }
    
    // Handle remainder
    for (int i = rem_start + tid; i < num_cols; i += num_threads) {
        local_max = fmaxf(local_max, row_in[i]);
    }
    
    // Warp-level reduction for max
    local_max = warp_reduce_max(local_max);
    
    // Store warp results to shared memory
    if (lane_id == 0) {
        smem[warp_id] = local_max;
    }
    __syncthreads();
    
    // Final reduction for max across warps
    if (tid < num_warps) {
        local_max = smem[tid];
    } else {
        local_max = -FLT_MAX;
    }
    if (tid < WARP_SIZE) {
        local_max = warp_reduce_max(local_max);
    }
    
    if (tid == 0) {
        smem[0] = local_max;
    }
    __syncthreads();
    float row_max = smem[0];
    
    // Phase 2: Compute sum of exp(x - max) using vectorized loads
    float local_sum = 0.0f;
    
    for (int i = tid; i < vec_cols; i += num_threads) {
        float4 v = load_float4(row_in + i * 4);
        local_sum += expf(v.x - row_max);
        local_sum += expf(v.y - row_max);
        local_sum += expf(v.z - row_max);
        local_sum += expf(v.w - row_max);
    }
    
    // Handle remainder
    for (int i = rem_start + tid; i < num_cols; i += num_threads) {
        local_sum += expf(row_in[i] - row_max);
    }
    
    // Warp-level reduction for sum
    local_sum = warp_reduce_sum(local_sum);
    
    if (lane_id == 0) {
        smem[warp_id] = local_sum;
    }
    __syncthreads();
    
    // Final reduction for sum
    if (tid < num_warps) {
        local_sum = smem[tid];
    } else {
        local_sum = 0.0f;
    }
    if (tid < WARP_SIZE) {
        local_sum = warp_reduce_sum(local_sum);
    }
    
    if (tid == 0) {
        smem[0] = local_sum;
    }
    __syncthreads();
    float inv_sum = 1.0f / smem[0];
    
    // Phase 3: Write normalized output with vectorized stores
    for (int i = tid; i < vec_cols; i += num_threads) {
        float4 v = load_float4(row_in + i * 4);
        float4 out;
        out.x = expf(v.x - row_max) * inv_sum;
        out.y = expf(v.y - row_max) * inv_sum;
        out.z = expf(v.z - row_max) * inv_sum;
        out.w = expf(v.w - row_max) * inv_sum;
        store_float4(row_out + i * 4, out);
    }
    
    // Handle remainder
    for (int i = rem_start + tid; i < num_cols; i += num_threads) {
        row_out[i] = expf(row_in[i] - row_max) * inv_sum;
    }
}

torch::Tensor softmax_hip(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(input.is_cuda(), "Input must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    
    int num_rows = input.size(0);
    int num_cols = input.size(1);
    
    auto output = torch::empty_like(input);
    
    // Use 1024 threads per block
    int block_size = 1024;
    int num_warps = block_size / WARP_SIZE;
    int shared_mem_size = num_warps * sizeof(float);
    
    dim3 grid(num_rows);
    dim3 block(block_size);
    
    softmax_kernel_vec4<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_rows, num_cols
    );
    
    return output;
}
"""

softmax_cpp_source = """
torch::Tensor softmax_hip(torch::Tensor input);
"""

softmax_module = load_inline(
    name="softmax_hip",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_hip_source,
    functions=["softmax_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax_op = softmax_module
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax_op.softmax_hip(x)


def get_inputs():
    x = torch.rand(4096, 393216).cuda()
    return [x]


def get_init_inputs():
    return []
