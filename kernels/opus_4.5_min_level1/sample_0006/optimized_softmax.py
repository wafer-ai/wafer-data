import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

softmax_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <float.h>

// Warp-level reduction for max
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 32; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down(val, offset));
    }
    return val;
}

// Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 32; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Online softmax kernel - each block handles one row
// Uses online algorithm to compute max and sum in single pass
__global__ void softmax_kernel(const float* __restrict__ input, 
                               float* __restrict__ output,
                               int dim) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    const float* row_input = input + row * dim;
    float* row_output = output + row * dim;
    
    // Phase 1: Find max using parallel reduction
    float thread_max = -FLT_MAX;
    for (int i = tid; i < dim; i += block_size) {
        thread_max = fmaxf(thread_max, row_input[i]);
    }
    
    // Warp-level reduction for max
    thread_max = warp_reduce_max(thread_max);
    
    // Block-level reduction using shared memory
    __shared__ float shared_max[32];  // One per warp
    __shared__ float shared_sum[32];
    
    int warp_id = tid / 64;
    int lane_id = tid % 64;
    
    if (lane_id == 0) {
        shared_max[warp_id] = thread_max;
    }
    __syncthreads();
    
    // First warp reduces across all warps
    if (tid < 32) {
        float val = (tid < (block_size + 63) / 64) ? shared_max[tid] : -FLT_MAX;
        val = warp_reduce_max(val);
        if (tid == 0) {
            shared_max[0] = val;
        }
    }
    __syncthreads();
    
    float row_max = shared_max[0];
    
    // Phase 2: Compute sum of exp(x - max)
    float thread_sum = 0.0f;
    for (int i = tid; i < dim; i += block_size) {
        thread_sum += expf(row_input[i] - row_max);
    }
    
    // Warp-level reduction for sum
    thread_sum = warp_reduce_sum(thread_sum);
    
    if (lane_id == 0) {
        shared_sum[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // First warp reduces across all warps
    if (tid < 32) {
        float val = (tid < (block_size + 63) / 64) ? shared_sum[tid] : 0.0f;
        val = warp_reduce_sum(val);
        if (tid == 0) {
            shared_sum[0] = val;
        }
    }
    __syncthreads();
    
    float row_sum = shared_sum[0];
    float inv_sum = 1.0f / row_sum;
    
    // Phase 3: Compute final softmax values
    for (int i = tid; i < dim; i += block_size) {
        output[row * dim + i] = expf(row_input[i] - row_max) * inv_sum;
    }
}

torch::Tensor softmax_hip(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
    
    auto output = torch::empty_like(input);
    
    int batch_size = input.size(0);
    int dim = input.size(1);
    
    // Use 1024 threads per block for large dimensions
    int block_size = 1024;
    
    softmax_kernel<<<batch_size, block_size>>>(
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
