import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# HIP kernel source
hip_source = """
#include <hip/hip_runtime.h>
#include <cmath>

// Fused Swish + Bias kernel
__global__ void swish_bias_kernel(const float* __restrict__ input,
                                   const float* __restrict__ bias,
                                   float* __restrict__ output,
                                   int batch_size,
                                   int features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * features;
    
    if (idx < total) {
        int feat_idx = idx % features;
        float x = input[idx];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        float swish = x * sigmoid_x;
        output[idx] = swish + bias[feat_idx];
    }
}

// GroupNorm kernel - compute mean and variance per group
__global__ void groupnorm_stats_kernel(const float* __restrict__ input,
                                        float* __restrict__ mean,
                                        float* __restrict__ var,
                                        int batch_size,
                                        int num_groups,
                                        int channels_per_group) {
    int batch_idx = blockIdx.x;
    int group_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    extern __shared__ float shared[];
    float* s_sum = shared;
    float* s_sum_sq = shared + blockDim.x;
    
    int group_start = group_idx * channels_per_group;
    int offset = batch_idx * num_groups * channels_per_group + group_start;
    
    // Each thread accumulates over multiple elements
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    for (int i = tid; i < channels_per_group; i += blockDim.x) {
        float val = input[offset + i];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();
    
    // Reduce in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        float m = s_sum[0] / channels_per_group;
        float v = s_sum_sq[0] / channels_per_group - m * m;
        mean[batch_idx * num_groups + group_idx] = m;
        var[batch_idx * num_groups + group_idx] = v;
    }
}

// GroupNorm kernel - normalize
__global__ void groupnorm_normalize_kernel(const float* __restrict__ input,
                                            const float* __restrict__ mean,
                                            const float* __restrict__ var,
                                            const float* __restrict__ gamma,
                                            const float* __restrict__ beta,
                                            float* __restrict__ output,
                                            int batch_size,
                                            int num_groups,
                                            int channels_per_group,
                                            float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_groups * channels_per_group;
    
    if (idx < total) {
        int batch_idx = idx / (num_groups * channels_per_group);
        int channel_idx = idx % (num_groups * channels_per_group);
        int group_idx = channel_idx / channels_per_group;
        
        float m = mean[batch_idx * num_groups + group_idx];
        float v = var[batch_idx * num_groups + group_idx];
        float inv_std = rsqrtf(v + eps);
        
        float x = input[idx];
        float normalized = (x - m) * inv_std;
        output[idx] = normalized * gamma[channel_idx] + beta[channel_idx];
    }
}
"""

# C++ wrapper source
cpp_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>

// Kernel declarations
__global__ void swish_bias_kernel(const float* input, const float* bias, float* output, int batch_size, int features);
__global__ void groupnorm_stats_kernel(const float* input, float* mean, float* var, int batch_size, int num_groups, int channels_per_group);
__global__ void groupnorm_normalize_kernel(const float* input, const float* mean, const float* var, const float* gamma, const float* beta, float* output, int batch_size, int num_groups, int channels_per_group, float eps);

torch::Tensor fused_swish_bias_groupnorm_hip(torch::Tensor input,
                                              torch::Tensor bias,
                                              torch::Tensor gamma,
                                              torch::Tensor beta,
                                              int num_groups,
                                              float eps) {
    int batch_size = input.size(0);
    int features = input.size(1);
    int channels_per_group = features / num_groups;
    
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto intermediate = torch::empty({batch_size, features}, options);
    auto output = torch::empty({batch_size, features}, options);
    auto mean = torch::empty({batch_size, num_groups}, options);
    auto var = torch::empty({batch_size, num_groups}, options);
    
    // Swish + Bias
    const int block_size = 256;
    int total_elements = batch_size * features;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    hipLaunchKernelGGL(swish_bias_kernel, dim3(num_blocks), dim3(block_size), 0, 0,
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        intermediate.data_ptr<float>(),
        batch_size,
        features
    );
    
    // GroupNorm stats
    dim3 stats_grid(batch_size, num_groups);
    int stats_threads = 64;  // channels_per_group is 64
    size_t shared_size = 2 * stats_threads * sizeof(float);
    
    hipLaunchKernelGGL(groupnorm_stats_kernel, stats_grid, dim3(stats_threads), shared_size, 0,
        intermediate.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        batch_size,
        num_groups,
        channels_per_group
    );
    
    // GroupNorm normalize
    hipLaunchKernelGGL(groupnorm_normalize_kernel, dim3(num_blocks), dim3(block_size), 0, 0,
        intermediate.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_groups,
        channels_per_group,
        eps
    );
    
    return output;
}
"""

fused_module = load_inline(
    name="fused_swish_bias_groupnorm",
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
