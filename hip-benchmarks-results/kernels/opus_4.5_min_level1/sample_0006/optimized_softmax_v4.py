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

// Online softmax: Track running max and sum in a single pass
// Formula: When new max is found, scale old sum: sum = sum * exp(old_max - new_max)
__device__ __forceinline__ void warp_reduce_online_softmax(float& max_val, float& sum_val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_max = __shfl_xor(max_val, offset);
        float other_sum = __shfl_xor(sum_val, offset);
        
        // Combine the two online softmax states
        if (other_max > max_val) {
            sum_val = sum_val * expf(max_val - other_max) + other_sum;
            max_val = other_max;
        } else {
            sum_val = sum_val + other_sum * expf(other_max - max_val);
        }
    }
}

// Single-kernel online softmax
__global__ void softmax_online_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int dim) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    const float* row_input = input + row * dim;
    float* row_output = output + row * dim;
    
    // Online softmax: single pass to compute max and sum
    float thread_max = -FLT_MAX;
    float thread_sum = 0.0f;
    
    int dim4 = dim / 4;
    const float4* row_input4 = reinterpret_cast<const float4*>(row_input);
    
    #pragma unroll 4
    for (int i = tid; i < dim4; i += block_size) {
        float4 v = row_input4[i];
        
        // Process each element in the float4
        float vals[4] = {v.x, v.y, v.z, v.w};
        
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float curr = vals[j];
            if (curr > thread_max) {
                thread_sum = thread_sum * expf(thread_max - curr) + 1.0f;
                thread_max = curr;
            } else {
                thread_sum += expf(curr - thread_max);
            }
        }
    }
    
    // Handle remaining elements
    for (int i = dim4 * 4 + tid; i < dim; i += block_size) {
        float curr = row_input[i];
        if (curr > thread_max) {
            thread_sum = thread_sum * expf(thread_max - curr) + 1.0f;
            thread_max = curr;
        } else {
            thread_sum += expf(curr - thread_max);
        }
    }
    
    // Warp-level reduction with online softmax
    warp_reduce_online_softmax(thread_max, thread_sum);
    
    // Block-level reduction
    __shared__ float shared_max[16];
    __shared__ float shared_sum[16];
    
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = block_size / WARP_SIZE;
    
    if (lane_id == 0) {
        shared_max[warp_id] = thread_max;
        shared_sum[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // First warp reduces across all warps
    if (tid < WARP_SIZE) {
        float max_val = (tid < num_warps) ? shared_max[tid] : -FLT_MAX;
        float sum_val = (tid < num_warps) ? shared_sum[tid] : 0.0f;
        
        warp_reduce_online_softmax(max_val, sum_val);
        
        if (tid == 0) {
            shared_max[0] = max_val;
            shared_sum[0] = sum_val;
        }
    }
    __syncthreads();
    
    float row_max = shared_max[0];
    float inv_sum = 1.0f / shared_sum[0];
    
    // Apply softmax with vectorized stores
    float4* row_output4 = reinterpret_cast<float4*>(row_output);
    
    #pragma unroll 4
    for (int i = tid; i < dim4; i += block_size) {
        float4 v = row_input4[i];
        float4 out;
        out.x = expf(v.x - row_max) * inv_sum;
        out.y = expf(v.y - row_max) * inv_sum;
        out.z = expf(v.z - row_max) * inv_sum;
        out.w = expf(v.w - row_max) * inv_sum;
        row_output4[i] = out;
    }
    
    // Handle remaining elements
    for (int i = dim4 * 4 + tid; i < dim; i += block_size) {
        row_output[i] = expf(row_input[i] - row_max) * inv_sum;
    }
}

torch::Tensor softmax_hip(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
    
    auto output = torch::empty_like(input);
    
    int batch_size = input.size(0);
    int dim = input.size(1);
    
    int block_size = 1024;
    
    softmax_online_kernel<<<batch_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        dim
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
