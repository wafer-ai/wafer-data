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
#define NUM_WARPS 16
#define BLOCK_SIZE (WARP_SIZE * NUM_WARPS)  // 1024 threads

// Online softmax warp reduction
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

// Vectorized online softmax with float4 loads
__global__ void softmax_kernel_online_vec4(const float* __restrict__ input, 
                                            float* __restrict__ output,
                                            int num_rows, int num_cols) {
    extern __shared__ char shared_mem[];
    float* s_max = reinterpret_cast<float*>(shared_mem);
    float* s_sum = s_max + NUM_WARPS;
    
    int row = blockIdx.x;
    if (row >= num_rows) return;
    
    const float* row_in = input + row * num_cols;
    float* row_out = output + row * num_cols;
    
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // Process most elements as float4
    int num_vec4 = num_cols / 4;
    int rem_start = num_vec4 * 4;
    
    // Online softmax reduction with vectorized loads
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    
    // Vectorized portion
    const float4* row_in_vec4 = reinterpret_cast<const float4*>(row_in);
    for (int i = tid; i < num_vec4; i += BLOCK_SIZE) {
        float4 vals = row_in_vec4[i];
        
        // Process all 4 elements
        float new_max = fmaxf(local_max, vals.x);
        local_sum = local_sum * expf(local_max - new_max) + expf(vals.x - new_max);
        local_max = new_max;
        
        new_max = fmaxf(local_max, vals.y);
        local_sum = local_sum * expf(local_max - new_max) + expf(vals.y - new_max);
        local_max = new_max;
        
        new_max = fmaxf(local_max, vals.z);
        local_sum = local_sum * expf(local_max - new_max) + expf(vals.z - new_max);
        local_max = new_max;
        
        new_max = fmaxf(local_max, vals.w);
        local_sum = local_sum * expf(local_max - new_max) + expf(vals.w - new_max);
        local_max = new_max;
    }
    
    // Remainder
    for (int i = rem_start + tid; i < num_cols; i += BLOCK_SIZE) {
        float val = row_in[i];
        float new_max = fmaxf(local_max, val);
        local_sum = local_sum * expf(local_max - new_max) + expf(val - new_max);
        local_max = new_max;
    }
    
    // Warp reduction
    warp_reduce_online(local_max, local_sum);
    
    if (lane_id == 0) {
        s_max[warp_id] = local_max;
        s_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    // Block reduction
    if (tid < NUM_WARPS) {
        local_max = s_max[tid];
        local_sum = s_sum[tid];
    } else {
        local_max = -FLT_MAX;
        local_sum = 0.0f;
    }
    
    if (tid < WARP_SIZE) {
        warp_reduce_online(local_max, local_sum);
    }
    
    if (tid == 0) {
        s_max[0] = local_max;
        s_sum[0] = local_sum;
    }
    __syncthreads();
    
    float row_max = s_max[0];
    float inv_sum = 1.0f / s_sum[0];
    
    // Write output with vectorized stores
    float4* row_out_vec4 = reinterpret_cast<float4*>(row_out);
    for (int i = tid; i < num_vec4; i += BLOCK_SIZE) {
        float4 vals = row_in_vec4[i];
        float4 out;
        out.x = expf(vals.x - row_max) * inv_sum;
        out.y = expf(vals.y - row_max) * inv_sum;
        out.z = expf(vals.z - row_max) * inv_sum;
        out.w = expf(vals.w - row_max) * inv_sum;
        row_out_vec4[i] = out;
    }
    
    // Remainder
    for (int i = rem_start + tid; i < num_cols; i += BLOCK_SIZE) {
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
    
    int shared_mem_size = 2 * NUM_WARPS * sizeof(float);
    
    softmax_kernel_online_vec4<<<num_rows, BLOCK_SIZE, shared_mem_size>>>(
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
