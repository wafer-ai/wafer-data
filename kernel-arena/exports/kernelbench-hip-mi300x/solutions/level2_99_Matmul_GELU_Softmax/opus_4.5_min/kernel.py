import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized fused GELU + Softmax kernel for MI300x
fused_gelu_softmax_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

#define WARP_SIZE 64
#define BLOCK_SIZE 512

__device__ __forceinline__ float gelu_fast(float x) {
    // Fast GELU approximation
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down(val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// 2-pass kernel with vectorized memory access
__global__ __launch_bounds__(BLOCK_SIZE)
void fused_gelu_softmax_kernel(const float* __restrict__ input, 
                                float* __restrict__ output,
                                const int features) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads = BLOCK_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = BLOCK_SIZE / WARP_SIZE;
    
    const float* row_in = input + row * features;
    float* row_out = output + row * features;
    
    __shared__ float shared_vals[num_warps];
    
    // Pass 1: Compute GELU and find max
    float local_max = -3.402823466e+38f;
    
    // Process 4 elements at a time for coalesced access
    for (int i = tid * 4; i < features; i += num_threads * 4) {
        if (i + 3 < features) {
            float4 v = *reinterpret_cast<const float4*>(row_in + i);
            float g0 = gelu_fast(v.x);
            float g1 = gelu_fast(v.y);
            float g2 = gelu_fast(v.z);
            float g3 = gelu_fast(v.w);
            
            // Store GELU values
            *reinterpret_cast<float4*>(row_out + i) = make_float4(g0, g1, g2, g3);
            
            local_max = fmaxf(local_max, fmaxf(fmaxf(g0, g1), fmaxf(g2, g3)));
        }
    }
    
    // Reduce max across warp
    local_max = warp_reduce_max(local_max);
    if (lane_id == 0) {
        shared_vals[warp_id] = local_max;
    }
    __syncthreads();
    
    // Final max reduction
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared_vals[lane_id] : -3.402823466e+38f;
        val = warp_reduce_max(val);
        if (lane_id == 0) {
            shared_vals[0] = val;
        }
    }
    __syncthreads();
    
    float row_max = shared_vals[0];
    
    // Pass 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    
    for (int i = tid * 4; i < features; i += num_threads * 4) {
        if (i + 3 < features) {
            float4 v = *reinterpret_cast<const float4*>(row_out + i);
            float e0 = expf(v.x - row_max);
            float e1 = expf(v.y - row_max);
            float e2 = expf(v.z - row_max);
            float e3 = expf(v.w - row_max);
            
            *reinterpret_cast<float4*>(row_out + i) = make_float4(e0, e1, e2, e3);
            local_sum += e0 + e1 + e2 + e3;
        }
    }
    
    // Reduce sum across warp
    local_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) {
        shared_vals[warp_id] = local_sum;
    }
    __syncthreads();
    
    // Final sum reduction
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared_vals[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            shared_vals[0] = val;
        }
    }
    __syncthreads();
    
    float inv_sum = 1.0f / shared_vals[0];
    
    // Pass 3: Normalize
    for (int i = tid * 4; i < features; i += num_threads * 4) {
        if (i + 3 < features) {
            float4 v = *reinterpret_cast<const float4*>(row_out + i);
            *reinterpret_cast<float4*>(row_out + i) = make_float4(
                v.x * inv_sum, v.y * inv_sum, v.z * inv_sum, v.w * inv_sum);
        }
    }
}

torch::Tensor fused_gelu_softmax_hip(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(input.size(1) % 4 == 0, "Features must be divisible by 4");
    
    const int batch_size = input.size(0);
    const int features = input.size(1);
    
    auto output = torch::empty_like(input);
    
    fused_gelu_softmax_kernel<<<batch_size, BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        features
    );
    
    return output;
}
"""

fused_gelu_softmax_cpp = """
torch::Tensor fused_gelu_softmax_hip(torch::Tensor input);
"""

fused_gelu_softmax = load_inline(
    name="fused_gelu_softmax",
    cpp_sources=fused_gelu_softmax_cpp,
    cuda_sources=fused_gelu_softmax_source,
    functions=["fused_gelu_softmax_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"],
)


class ModelNew(nn.Module):
    """
    Optimized model with fused GELU + Softmax kernel.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.fused_gelu_softmax = fused_gelu_softmax

    def forward(self, x):
        x = self.linear(x)
        x = self.fused_gelu_softmax.fused_gelu_softmax_hip(x)
        return x


def get_inputs():
    return [torch.rand(1024, 8192).cuda()]


def get_init_inputs():
    return [8192, 8192]
