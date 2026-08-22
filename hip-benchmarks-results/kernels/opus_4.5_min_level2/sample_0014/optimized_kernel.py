import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused GELU + Softmax kernel
fused_gelu_softmax_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>

#define WARP_SIZE 64
#define BLOCK_SIZE 256

__device__ __forceinline__ float gelu(float x) {
    // GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Warp reduction for max
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down(val, offset));
    }
    return val;
}

// Warp reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__global__ void fused_gelu_softmax_kernel(const float* __restrict__ input, 
                                           float* __restrict__ output,
                                           int batch_size, int features) {
    // Each block handles one row
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    const float* row_in = input + row * features;
    float* row_out = output + row * features;
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = num_threads / WARP_SIZE;
    
    __shared__ float shared_max[BLOCK_SIZE / WARP_SIZE];
    __shared__ float shared_sum[BLOCK_SIZE / WARP_SIZE];
    
    // First pass: compute GELU and find max
    float local_max = -INFINITY;
    for (int i = tid; i < features; i += num_threads) {
        float val = gelu(row_in[i]);
        local_max = fmaxf(local_max, val);
    }
    
    // Warp reduce max
    local_max = warp_reduce_max(local_max);
    if (lane_id == 0) {
        shared_max[warp_id] = local_max;
    }
    __syncthreads();
    
    // Final reduction for max across warps
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared_max[lane_id] : -INFINITY;
        val = warp_reduce_max(val);
        if (lane_id == 0) {
            shared_max[0] = val;
        }
    }
    __syncthreads();
    
    float row_max = shared_max[0];
    
    // Second pass: compute exp(gelu(x) - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < features; i += num_threads) {
        float val = gelu(row_in[i]);
        float exp_val = expf(val - row_max);
        row_out[i] = exp_val;  // Store temporarily
        local_sum += exp_val;
    }
    
    // Warp reduce sum
    local_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) {
        shared_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    // Final reduction for sum across warps
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            shared_sum[0] = val;
        }
    }
    __syncthreads();
    
    float row_sum = shared_sum[0];
    float inv_sum = 1.0f / row_sum;
    
    // Third pass: normalize
    for (int i = tid; i < features; i += num_threads) {
        row_out[i] *= inv_sum;
    }
}

torch::Tensor fused_gelu_softmax_hip(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    
    int batch_size = input.size(0);
    int features = input.size(1);
    
    auto output = torch::empty_like(input);
    
    dim3 grid(batch_size);
    dim3 block(BLOCK_SIZE);
    
    fused_gelu_softmax_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
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
