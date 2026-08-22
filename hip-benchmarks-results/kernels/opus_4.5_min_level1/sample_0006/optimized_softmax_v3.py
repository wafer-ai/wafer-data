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

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Two-kernel approach: First pass finds max+sum, second pass computes output
__global__ void softmax_reduce_kernel(const float* __restrict__ input,
                                       float* __restrict__ row_max,
                                       float* __restrict__ row_sum,
                                       int dim) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    const float* row_input = input + row * dim;
    
    // Phase 1: Find max
    float thread_max = -FLT_MAX;
    
    int dim4 = dim / 4;
    const float4* row_input4 = reinterpret_cast<const float4*>(row_input);
    
    #pragma unroll 4
    for (int i = tid; i < dim4; i += block_size) {
        float4 v = row_input4[i];
        thread_max = fmaxf(thread_max, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    }
    
    for (int i = dim4 * 4 + tid; i < dim; i += block_size) {
        thread_max = fmaxf(thread_max, row_input[i]);
    }
    
    thread_max = warp_reduce_max(thread_max);
    
    __shared__ float shared_data[16];
    
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = block_size / WARP_SIZE;
    
    if (lane_id == 0) {
        shared_data[warp_id] = thread_max;
    }
    __syncthreads();
    
    if (tid < WARP_SIZE) {
        float val = (tid < num_warps) ? shared_data[tid] : -FLT_MAX;
        val = warp_reduce_max(val);
        if (tid == 0) {
            shared_data[0] = val;
        }
    }
    __syncthreads();
    
    float max_val = shared_data[0];
    
    // Phase 2: Compute sum of exp(x - max)
    float thread_sum = 0.0f;
    
    #pragma unroll 4
    for (int i = tid; i < dim4; i += block_size) {
        float4 v = row_input4[i];
        thread_sum += expf(v.x - max_val) + expf(v.y - max_val) + 
                      expf(v.z - max_val) + expf(v.w - max_val);
    }
    
    for (int i = dim4 * 4 + tid; i < dim; i += block_size) {
        thread_sum += expf(row_input[i] - max_val);
    }
    
    thread_sum = warp_reduce_sum(thread_sum);
    
    if (lane_id == 0) {
        shared_data[warp_id] = thread_sum;
    }
    __syncthreads();
    
    if (tid < WARP_SIZE) {
        float val = (tid < num_warps) ? shared_data[tid] : 0.0f;
        val = warp_reduce_sum(val);
        if (tid == 0) {
            row_max[row] = max_val;
            row_sum[row] = val;
        }
    }
}

__global__ void softmax_apply_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      const float* __restrict__ row_max,
                                      const float* __restrict__ row_sum,
                                      int dim) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    float max_val = row_max[row];
    float inv_sum = 1.0f / row_sum[row];
    
    const float* row_input = input + row * dim;
    float* row_output = output + row * dim;
    
    int dim4 = dim / 4;
    const float4* row_input4 = reinterpret_cast<const float4*>(row_input);
    float4* row_output4 = reinterpret_cast<float4*>(row_output);
    
    #pragma unroll 4
    for (int i = tid; i < dim4; i += block_size) {
        float4 v = row_input4[i];
        float4 out;
        out.x = expf(v.x - max_val) * inv_sum;
        out.y = expf(v.y - max_val) * inv_sum;
        out.z = expf(v.z - max_val) * inv_sum;
        out.w = expf(v.w - max_val) * inv_sum;
        row_output4[i] = out;
    }
    
    for (int i = dim4 * 4 + tid; i < dim; i += block_size) {
        row_output[i] = expf(row_input[i] - max_val) * inv_sum;
    }
}

torch::Tensor softmax_hip(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
    
    auto output = torch::empty_like(input);
    
    int batch_size = input.size(0);
    int dim = input.size(1);
    
    auto row_max = torch::empty({batch_size}, input.options());
    auto row_sum = torch::empty({batch_size}, input.options());
    
    int block_size = 1024;
    
    softmax_reduce_kernel<<<batch_size, block_size>>>(
        input.data_ptr<float>(),
        row_max.data_ptr<float>(),
        row_sum.data_ptr<float>(),
        dim
    );
    
    softmax_apply_kernel<<<batch_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        row_max.data_ptr<float>(),
        row_sum.data_ptr<float>(),
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
