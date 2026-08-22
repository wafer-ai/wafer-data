import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused kernel with one pass using registers
fused_kernel_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

#define CHANNELS_PER_GROUP 64
#define BLOCK_SIZE 64

// Welford's online algorithm for numerically stable variance
__device__ __forceinline__ void welford_combine(float& mean, float& m2, float& count,
                                                  float val) {
    count += 1.0f;
    float delta = val - mean;
    mean += delta / count;
    float delta2 = val - mean;
    m2 += delta * delta2;
}

// Optimized kernel where each thread handles exactly one element
// channels_per_group = 64, block_size = 64
__global__ void fused_swish_bias_groupnorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int batch_size,
    int out_features,
    int num_groups,
    float eps) {
    
    const int channels_per_group = CHANNELS_PER_GROUP;
    
    int batch_idx = blockIdx.x / num_groups;
    int group_idx = blockIdx.x % num_groups;
    
    int group_start = group_idx * channels_per_group;
    int base_offset = batch_idx * out_features + group_start;
    
    // Each thread handles one element
    int local_idx = threadIdx.x;
    int idx = base_offset + local_idx;
    int feature_idx = group_start + local_idx;
    
    // Compute Swish + bias
    float x = input[idx];
    float sigmoid_x = 1.0f / (1.0f + expf(-x));
    float val = sigmoid_x * x + bias[feature_idx];
    
    // Shared memory for reduction
    __shared__ float shared_sum[BLOCK_SIZE];
    __shared__ float shared_sq_sum[BLOCK_SIZE];
    
    shared_sum[local_idx] = val;
    shared_sq_sum[local_idx] = val * val;
    __syncthreads();
    
    // Parallel reduction
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (local_idx < stride) {
            shared_sum[local_idx] += shared_sum[local_idx + stride];
            shared_sq_sum[local_idx] += shared_sq_sum[local_idx + stride];
        }
        __syncthreads();
    }
    
    // Compute mean and variance
    float mean = shared_sum[0] / channels_per_group;
    float variance = shared_sq_sum[0] / channels_per_group - mean * mean;
    float inv_std = rsqrtf(variance + eps);
    
    // Normalize and apply gamma/beta
    float normalized = (val - mean) * inv_std;
    output[idx] = normalized * gamma[feature_idx] + beta[feature_idx];
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
    
    auto output = torch::empty_like(input);
    
    // One block per (batch, group) pair
    int num_blocks = batch_size * num_groups;
    
    fused_swish_bias_groupnorm_kernel<<<num_blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features,
        num_groups,
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
