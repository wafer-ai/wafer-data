import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# HIP kernel source - higher parallelism version
hip_source = """
#include <hip/hip_runtime.h>
#include <cmath>

#define WARP_SIZE 64

// Warp reduce sum using shuffle
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Fused kernel with 256 threads per block (4 groups per block)
// Each warp (64 threads) handles one group
__global__ __launch_bounds__(256) void fused_swish_bias_groupnorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int batch_size,
    int features,
    int num_groups,
    int channels_per_group,
    float eps) {
    
    // Block handles one batch element, with multiple groups
    int batch_idx = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;  // which warp in this block
    int lane_id = threadIdx.x % WARP_SIZE;  // position within warp
    
    // Each block handles 4 warps = 4 groups (since channels_per_group=64=WARP_SIZE)
    int groups_per_block = blockDim.x / WARP_SIZE;
    int base_group = blockIdx.y * groups_per_block;
    int group_idx = base_group + warp_id;
    
    if (group_idx >= num_groups) return;
    
    int group_start = group_idx * channels_per_group;
    int offset = batch_idx * features + group_start;
    int feat_idx = group_start + lane_id;
    
    // Step 1: Compute swish + bias
    float val = 0.0f;
    if (lane_id < channels_per_group) {
        float x = input[offset + lane_id];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        val = x * sigmoid_x + bias[feat_idx];
    }
    
    // Step 2: Compute mean and variance using warp reduction
    float sum = warpReduceSum(val);
    float sum_sq = warpReduceSum(val * val);
    
    // All threads in warp have same result now
    float mean = sum / channels_per_group;
    float variance = sum_sq / channels_per_group - mean * mean;
    float inv_std = rsqrtf(variance + eps);
    
    // Step 3: Normalize and write output
    if (lane_id < channels_per_group) {
        float normalized = (val - mean) * inv_std;
        output[offset + lane_id] = normalized * gamma[feat_idx] + beta[feat_idx];
    }
}
"""

# C++ wrapper source
cpp_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>

__global__ void fused_swish_bias_groupnorm_kernel(
    const float* input, const float* bias, const float* gamma, const float* beta,
    float* output, int batch_size, int features, int num_groups,
    int channels_per_group, float eps);

torch::Tensor fused_swish_bias_groupnorm_hip(torch::Tensor input,
                                              torch::Tensor bias,
                                              torch::Tensor gamma,
                                              torch::Tensor beta,
                                              int num_groups,
                                              float eps) {
    int batch_size = input.size(0);
    int features = input.size(1);
    int channels_per_group = features / num_groups;
    
    auto output = torch::empty_like(input);
    
    // 256 threads = 4 warps, each warp handles one group
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 64;  // 4 warps
    const int groups_per_block = warps_per_block;  // 4 groups per block
    int blocks_y = (num_groups + groups_per_block - 1) / groups_per_block;
    
    dim3 grid(batch_size, blocks_y);
    dim3 block(threads_per_block);
    
    hipLaunchKernelGGL(fused_swish_bias_groupnorm_kernel, grid, block, 0, 0,
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        features,
        num_groups,
        channels_per_group,
        eps
    );
    
    return output;
}
"""

fused_module = load_inline(
    name="fused_swish_bias_groupnorm_v4",
    cpp_sources=cpp_source,
    cuda_sources=hip_source,
    functions=["fused_swish_bias_groupnorm_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized model with fused Swish + Bias + GroupNorm kernel.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.num_groups = num_groups
        # GroupNorm parameters
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))
        self.eps = 1e-5
        self.fused_module = fused_module

    def forward(self, x):
        x = self.matmul(x)
        x = self.fused_module.fused_swish_bias_groupnorm_hip(
            x, self.bias, self.gamma, self.beta, self.num_groups, self.eps
        )
        return x


def custom_kernel(inputs):
    # Create model with same architecture
    in_features = 1024
    out_features = 4096
    num_groups = 64
    bias_shape = (out_features,)
    
    model = ModelNew(in_features, out_features, num_groups, bias_shape).cuda()
    return model(inputs[0])
