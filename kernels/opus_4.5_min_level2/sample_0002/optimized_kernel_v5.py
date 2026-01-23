import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused kernel: Swish + bias + GroupNorm 
# Two-pass approach using global memory (L2 cache friendly)
fused_kernel_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

// Fused Swish + Bias + GroupNorm kernel
// Each block handles one (batch, group) pair
// Two-pass: first computes swish+bias and stats, second normalizes
__global__ void fused_swish_bias_groupnorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int batch_size,
    int out_features,
    int num_groups,
    int channels_per_group,
    float eps) {
    
    int batch_idx = blockIdx.x / num_groups;
    int group_idx = blockIdx.x % num_groups;
    
    int group_start = group_idx * channels_per_group;
    int base_offset = batch_idx * out_features + group_start;
    
    extern __shared__ float shared_mem[];
    float* shared_sum = shared_mem;
    float* shared_sq_sum = shared_mem + blockDim.x;
    
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    
    // First pass: compute swish + bias and write to output, accumulate stats
    for (int i = threadIdx.x; i < channels_per_group; i += blockDim.x) {
        int idx = base_offset + i;
        int feature_idx = group_start + i;
        
        float x = input[idx];
        // Swish: sigmoid(x) * x
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        float val = sigmoid_x * x + bias[feature_idx];
        
        // Write to output (will be read back in second pass - L2 cached)
        output[idx] = val;
        
        local_sum += val;
        local_sq_sum += val * val;
    }
    
    // Store local sums to shared memory
    shared_sum[threadIdx.x] = local_sum;
    shared_sq_sum[threadIdx.x] = local_sq_sum;
    __syncthreads();
    
    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
            shared_sq_sum[threadIdx.x] += shared_sq_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Compute mean and variance
    float mean = shared_sum[0] / channels_per_group;
    float variance = shared_sq_sum[0] / channels_per_group - mean * mean;
    float inv_std = rsqrtf(variance + eps);
    
    __syncthreads();
    
    // Second pass: read from output, normalize, write back
    for (int i = threadIdx.x; i < channels_per_group; i += blockDim.x) {
        int idx = base_offset + i;
        int feature_idx = group_start + i;
        
        float val = output[idx];
        float normalized = (val - mean) * inv_std;
        output[idx] = normalized * gamma[feature_idx] + beta[feature_idx];
    }
}

torch::Tensor fused_swish_bias_groupnorm_hip(
    torch::Tensor input,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float eps) {
    
    int batch_size = input.size(0);
    int out_features = input.size(1);
    int channels_per_group = out_features / num_groups;
    
    auto output = torch::empty_like(input);
    
    // One block per (batch, group) pair
    int num_blocks = batch_size * num_groups;
    
    // Choose block size based on channels_per_group
    int block_size = 64;  // channels_per_group = 64 for this problem
    
    size_t shared_mem_size = 2 * block_size * sizeof(float);
    
    fused_swish_bias_groupnorm_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features,
        num_groups,
        channels_per_group,
        eps);
    
    return output;
}
"""

fused_kernel = load_inline(
    name="fused_kernel",
    cpp_sources=fused_kernel_cpp_source,
    functions=["fused_swish_bias_groupnorm_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized model with fully fused Swish + bias + GroupNorm kernel.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.num_groups = num_groups
        self.fused_kernel = fused_kernel

    def forward(self, x):
        x = self.matmul(x)
        # Fully fused: Swish + bias + GroupNorm
        x = self.fused_kernel.fused_swish_bias_groupnorm_hip(
            x, 
            self.bias,
            self.group_norm.weight,
            self.group_norm.bias,
            self.num_groups,
            self.group_norm.eps)
        return x


def get_inputs():
    return [torch.rand(32768, 1024).cuda()]


def get_init_inputs():
    return [1024, 4096, 64, (4096,)]
