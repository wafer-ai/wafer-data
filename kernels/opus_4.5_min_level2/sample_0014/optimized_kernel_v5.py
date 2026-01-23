import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Online softmax with fused GELU - single pass for max and exp sum
fused_gelu_softmax_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

#define WARP_SIZE 64
#define BLOCK_SIZE 256

__device__ __forceinline__ float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down(val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Online softmax: compute max and sum in a single pass using the formula:
// When we see a new max m', we update sum: sum' = sum * exp(m - m') + exp(x - m')
__global__ void fused_gelu_softmax_online_kernel(const float4* __restrict__ input, 
                                                   float4* __restrict__ output,
                                                   int batch_size, int vec_features) {
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    const float4* row_in = input + row * vec_features;
    float4* row_out = output + row * vec_features;
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = num_threads / WARP_SIZE;
    
    __shared__ float shared_max[BLOCK_SIZE / WARP_SIZE];
    __shared__ float shared_sum[BLOCK_SIZE / WARP_SIZE];
    
    // Online softmax: process elements and track max + exp-sum together
    float local_max = -3.402823466e+38f;
    float local_sum = 0.0f;
    
    // First pass: Online softmax (compute max and running exp-sum together)
    for (int i = tid; i < vec_features; i += num_threads) {
        float4 v = row_in[i];
        float g0 = gelu(v.x);
        float g1 = gelu(v.y);
        float g2 = gelu(v.z);
        float g3 = gelu(v.w);
        
        // Store GELU values
        row_out[i] = make_float4(g0, g1, g2, g3);
        
        // Online update for each element
        // For g0
        float new_max = fmaxf(local_max, g0);
        local_sum = local_sum * expf(local_max - new_max) + expf(g0 - new_max);
        local_max = new_max;
        
        // For g1
        new_max = fmaxf(local_max, g1);
        local_sum = local_sum * expf(local_max - new_max) + expf(g1 - new_max);
        local_max = new_max;
        
        // For g2
        new_max = fmaxf(local_max, g2);
        local_sum = local_sum * expf(local_max - new_max) + expf(g2 - new_max);
        local_max = new_max;
        
        // For g3
        new_max = fmaxf(local_max, g3);
        local_sum = local_sum * expf(local_max - new_max) + expf(g3 - new_max);
        local_max = new_max;
    }
    
    // Share local max/sum for cross-warp reduction
    if (lane_id == 0) {
        shared_max[warp_id] = local_max;
        shared_sum[warp_id] = local_sum;
    }
    
    // First reduce within warp
    float warp_max = warp_reduce_max(local_max);
    float warp_sum = local_sum * expf(local_max - warp_max);
    warp_sum = warp_reduce_sum(warp_sum);
    
    if (lane_id == 0) {
        shared_max[warp_id] = warp_max;
        shared_sum[warp_id] = warp_sum;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (warp_id == 0) {
        float m = (lane_id < num_warps) ? shared_max[lane_id] : -3.402823466e+38f;
        float s = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
        
        // Find global max
        float global_max = warp_reduce_max(m);
        
        // Adjust sum based on global max
        float adjusted_sum = s * expf(m - global_max);
        float global_sum = warp_reduce_sum(adjusted_sum);
        
        if (lane_id == 0) {
            shared_max[0] = global_max;
            shared_sum[0] = global_sum;
        }
    }
    __syncthreads();
    
    float row_max = shared_max[0];
    float inv_sum = 1.0f / shared_sum[0];
    
    // Second pass: normalize using stored GELU values
    for (int i = tid; i < vec_features; i += num_threads) {
        float4 gelu_val = row_out[i];
        float4 result;
        result.x = expf(gelu_val.x - row_max) * inv_sum;
        result.y = expf(gelu_val.y - row_max) * inv_sum;
        result.z = expf(gelu_val.z - row_max) * inv_sum;
        result.w = expf(gelu_val.w - row_max) * inv_sum;
        row_out[i] = result;
    }
}

torch::Tensor fused_gelu_softmax_hip(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(input.size(1) % 4 == 0, "Features must be divisible by 4");
    
    int batch_size = input.size(0);
    int features = input.size(1);
    int vec_features = features / 4;
    
    auto output = torch::empty_like(input);
    
    dim3 grid(batch_size);
    dim3 block(BLOCK_SIZE);
    
    fused_gelu_softmax_online_kernel<<<grid, block>>>(
        reinterpret_cast<const float4*>(input.data_ptr<float>()),
        reinterpret_cast<float4*>(output.data_ptr<float>()),
        batch_size,
        vec_features
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
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized model with fused GELU + Softmax kernel using online softmax.
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
