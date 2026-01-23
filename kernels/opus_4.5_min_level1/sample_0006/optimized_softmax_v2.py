import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

softmax_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <float.h>

// Wavefront size on AMD is 64
#define WARP_SIZE 64

// Warp-level reduction for max using AMD's 64-wide wavefronts
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

// Warp-level reduction for sum using AMD's 64-wide wavefronts
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Optimized softmax kernel using vectorized loads
__global__ void softmax_kernel_v2(const float* __restrict__ input, 
                                   float* __restrict__ output,
                                   int dim) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    const float* row_input = input + row * dim;
    float* row_output = output + row * dim;
    
    // Phase 1: Find max using parallel reduction with vectorized loads
    float thread_max = -FLT_MAX;
    
    // Process 4 elements at a time using float4
    int dim4 = dim / 4;
    const float4* row_input4 = reinterpret_cast<const float4*>(row_input);
    
    for (int i = tid; i < dim4; i += block_size) {
        float4 v = row_input4[i];
        thread_max = fmaxf(thread_max, v.x);
        thread_max = fmaxf(thread_max, v.y);
        thread_max = fmaxf(thread_max, v.z);
        thread_max = fmaxf(thread_max, v.w);
    }
    
    // Handle remaining elements
    for (int i = dim4 * 4 + tid; i < dim; i += block_size) {
        thread_max = fmaxf(thread_max, row_input[i]);
    }
    
    // Warp-level reduction for max
    thread_max = warp_reduce_max(thread_max);
    
    // Block-level reduction using shared memory
    __shared__ float shared_data[16];  // One per warp (1024/64 = 16 warps max)
    
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = block_size / WARP_SIZE;
    
    if (lane_id == 0) {
        shared_data[warp_id] = thread_max;
    }
    __syncthreads();
    
    // First warp reduces across all warps
    if (tid < WARP_SIZE) {
        float val = (tid < num_warps) ? shared_data[tid] : -FLT_MAX;
        val = warp_reduce_max(val);
        if (tid == 0) {
            shared_data[0] = val;
        }
    }
    __syncthreads();
    
    float row_max = shared_data[0];
    
    // Phase 2: Compute sum of exp(x - max) with vectorized loads
    float thread_sum = 0.0f;
    
    for (int i = tid; i < dim4; i += block_size) {
        float4 v = row_input4[i];
        thread_sum += expf(v.x - row_max);
        thread_sum += expf(v.y - row_max);
        thread_sum += expf(v.z - row_max);
        thread_sum += expf(v.w - row_max);
    }
    
    // Handle remaining elements
    for (int i = dim4 * 4 + tid; i < dim; i += block_size) {
        thread_sum += expf(row_input[i] - row_max);
    }
    
    // Warp-level reduction for sum
    thread_sum = warp_reduce_sum(thread_sum);
    
    if (lane_id == 0) {
        shared_data[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // First warp reduces across all warps
    if (tid < WARP_SIZE) {
        float val = (tid < num_warps) ? shared_data[tid] : 0.0f;
        val = warp_reduce_sum(val);
        if (tid == 0) {
            shared_data[0] = val;
        }
    }
    __syncthreads();
    
    float inv_sum = 1.0f / shared_data[0];
    
    // Phase 3: Compute final softmax values with vectorized stores
    float4* row_output4 = reinterpret_cast<float4*>(row_output);
    
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
    
    // Use 1024 threads per block
    int block_size = 1024;
    
    softmax_kernel_v2<<<batch_size, block_size>>>(
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
