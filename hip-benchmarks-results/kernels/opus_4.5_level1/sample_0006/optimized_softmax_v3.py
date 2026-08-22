import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

softmax_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <float.h>

#define WARP_SIZE 64

// Online softmax: combines max-finding and sum computation in a single pass
// For two partial results (m1, d1) and (m2, d2) where m=max, d=sum of exp(x-m):
// m_new = max(m1, m2)
// d_new = d1 * exp(m1 - m_new) + d2 * exp(m2 - m_new)

__device__ __forceinline__ void warp_reduce_online(float& max_val, float& sum_val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_max = __shfl_xor(max_val, offset);
        float other_sum = __shfl_xor(sum_val, offset);
        
        float new_max = fmaxf(max_val, other_max);
        sum_val = sum_val * expf(max_val - new_max) + other_sum * expf(other_max - new_max);
        max_val = new_max;
    }
}

// Two-pass online softmax: 
// Pass 1: Compute (max, sum) using online algorithm
// Pass 2: Write normalized output
__global__ void softmax_kernel_online(const float* __restrict__ input, 
                                       float* __restrict__ output,
                                       int num_rows, int num_cols) {
    extern __shared__ char shared_mem[];
    float* s_max = reinterpret_cast<float*>(shared_mem);
    float* s_sum = s_max + (blockDim.x / WARP_SIZE);
    
    int row = blockIdx.x;
    if (row >= num_rows) return;
    
    const float* row_in = input + row * num_cols;
    float* row_out = output + row * num_cols;
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int num_warps = num_threads / WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // Phase 1: Online softmax reduction to get max and sum simultaneously
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    
    for (int i = tid; i < num_cols; i += num_threads) {
        float val = row_in[i];
        float new_max = fmaxf(local_max, val);
        local_sum = local_sum * expf(local_max - new_max) + expf(val - new_max);
        local_max = new_max;
    }
    
    // Warp-level online reduction
    warp_reduce_online(local_max, local_sum);
    
    // Store warp results
    if (lane_id == 0) {
        s_max[warp_id] = local_max;
        s_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    // Final reduction across warps (done by first warp)
    if (tid < num_warps) {
        local_max = s_max[tid];
        local_sum = s_sum[tid];
    } else {
        local_max = -FLT_MAX;
        local_sum = 0.0f;
    }
    
    if (tid < WARP_SIZE) {
        warp_reduce_online(local_max, local_sum);
    }
    
    // Broadcast results
    if (tid == 0) {
        s_max[0] = local_max;
        s_sum[0] = local_sum;
    }
    __syncthreads();
    
    float row_max = s_max[0];
    float inv_sum = 1.0f / s_sum[0];
    
    // Phase 2: Write normalized output
    for (int i = tid; i < num_cols; i += num_threads) {
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
    int shared_mem_size = 2 * num_warps * sizeof(float);
    
    dim3 grid(num_rows);
    dim3 block(block_size);
    
    softmax_kernel_online<<<grid, block, shared_mem_size>>>(
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
