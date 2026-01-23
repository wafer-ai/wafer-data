import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# HIP kernel source - more optimized version
hip_source = """
#include <hip/hip_runtime.h>
#include <cmath>

#define WARP_SIZE 64
#define BLOCK_SIZE 256

// Warp reduce sum
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Block reduce sum
__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warpReduceSum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    // Only first warp does final reduction
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

// Fused Swish + Bias + GroupNorm stats kernel
// Each block handles one group for one batch element
__global__ void swish_bias_groupnorm_stats_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ intermediate,
    float* __restrict__ mean,
    float* __restrict__ var,
    int batch_size,
    int features,
    int num_groups,
    int channels_per_group) {
    
    int batch_idx = blockIdx.x;
    int group_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    int group_start = group_idx * channels_per_group;
    int offset = batch_idx * features + group_start;
    
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    // Process elements in this group
    for (int i = tid; i < channels_per_group; i += blockDim.x) {
        int feat_idx = group_start + i;
        float x = input[offset + i];
        
        // Swish activation
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        float swish = x * sigmoid_x;
        
        // Add bias
        float val = swish + bias[feat_idx];
        intermediate[offset + i] = val;
        
        // Accumulate stats
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    // Reduce within block
    float sum = blockReduceSum(local_sum);
    float sum_sq = blockReduceSum(local_sum_sq);
    
    if (tid == 0) {
        float m = sum / channels_per_group;
        float v = sum_sq / channels_per_group - m * m;
        mean[batch_idx * num_groups + group_idx] = m;
        var[batch_idx * num_groups + group_idx] = v;
    }
}

// GroupNorm normalize kernel - optimized with vectorized access
__global__ void groupnorm_normalize_kernel(
    const float* __restrict__ input,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int batch_size,
    int features,
    int num_groups,
    int channels_per_group,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * features;
    
    // Process 4 elements at a time if possible
    int idx4 = idx * 4;
    if (idx4 + 3 < total) {
        int batch_idx0 = idx4 / features;
        int channel_idx0 = idx4 % features;
        
        // Check if all 4 are in same batch and group
        int batch_idx3 = (idx4 + 3) / features;
        
        if (batch_idx0 == batch_idx3) {
            int group_idx0 = channel_idx0 / channels_per_group;
            int group_idx3 = ((idx4 + 3) % features) / channels_per_group;
            
            if (group_idx0 == group_idx3) {
                // All 4 elements share same mean/var
                float m = mean[batch_idx0 * num_groups + group_idx0];
                float v = var[batch_idx0 * num_groups + group_idx0];
                float inv_std = rsqrtf(v + eps);
                
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int global_idx = idx4 + i;
                    int chan = global_idx % features;
                    float x = input[global_idx];
                    float normalized = (x - m) * inv_std;
                    output[global_idx] = normalized * gamma[chan] + beta[chan];
                }
                return;
            }
        }
    }
    
    // Fallback for remaining elements
    for (int i = 0; i < 4 && idx4 + i < total; i++) {
        int global_idx = idx4 + i;
        int batch_idx = global_idx / features;
        int channel_idx = global_idx % features;
        int group_idx = channel_idx / channels_per_group;
        
        float m = mean[batch_idx * num_groups + group_idx];
        float v = var[batch_idx * num_groups + group_idx];
        float inv_std = rsqrtf(v + eps);
        
        float x = input[global_idx];
        float normalized = (x - m) * inv_std;
        output[global_idx] = normalized * gamma[channel_idx] + beta[channel_idx];
    }
}

// Simple normalize kernel (no vectorization for cleaner code)
__global__ void groupnorm_normalize_simple_kernel(
    const float* __restrict__ input,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int batch_size,
    int features,
    int num_groups,
    int channels_per_group,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * features;
    
    if (idx < total) {
        int batch_idx = idx / features;
        int channel_idx = idx % features;
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
__global__ void swish_bias_groupnorm_stats_kernel(
    const float* input, const float* bias, float* intermediate,
    float* mean, float* var, int batch_size, int features,
    int num_groups, int channels_per_group);
    
__global__ void groupnorm_normalize_simple_kernel(
    const float* input, const float* mean, const float* var,
    const float* gamma, const float* beta, float* output,
    int batch_size, int features, int num_groups,
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
    
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto intermediate = torch::empty({batch_size, features}, options);
    auto output = torch::empty({batch_size, features}, options);
    auto mean = torch::empty({batch_size, num_groups}, options);
    auto var = torch::empty({batch_size, num_groups}, options);
    
    // Fused Swish + Bias + Stats - one block per (batch, group)
    dim3 stats_grid(batch_size, num_groups);
    int stats_threads = 64;  // channels_per_group is 64
    
    hipLaunchKernelGGL(swish_bias_groupnorm_stats_kernel, stats_grid, dim3(stats_threads), 0, 0,
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        intermediate.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        batch_size,
        features,
        num_groups,
        channels_per_group
    );
    
    // GroupNorm normalize
    const int block_size = 256;
    int total_elements = batch_size * features;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    hipLaunchKernelGGL(groupnorm_normalize_simple_kernel, dim3(num_blocks), dim3(block_size), 0, 0,
        intermediate.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
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
    name="fused_swish_bias_groupnorm_v2",
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
