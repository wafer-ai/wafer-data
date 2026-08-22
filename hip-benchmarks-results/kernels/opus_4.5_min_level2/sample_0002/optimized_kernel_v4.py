import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Highly optimized fused kernel: Swish + bias + GroupNorm with better occupancy
fused_kernel_cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

#define WARP_SIZE 64

// Fused Swish + Bias + GroupNorm kernel with better memory access
// Each block handles one (batch, group) pair
// Uses larger block size and vectorized loads where possible
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
    float* shared_data = shared_mem + 2 * blockDim.x; // Store intermediate values
    
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    
    // Process 4 elements per thread if possible
    int num_vec4 = channels_per_group / 4;
    int remainder_start = num_vec4 * 4;
    
    // First pass: compute swish + bias and accumulate for mean/variance
    // Use float4 vectorized loads
    const float4* input_vec = reinterpret_cast<const float4*>(input + base_offset);
    const float4* bias_vec = reinterpret_cast<const float4*>(bias + group_start);
    
    for (int i = threadIdx.x; i < num_vec4; i += blockDim.x) {
        float4 x = input_vec[i];
        float4 b = bias_vec[i];
        
        // Swish + bias for each component
        float v0 = (1.0f / (1.0f + expf(-x.x))) * x.x + b.x;
        float v1 = (1.0f / (1.0f + expf(-x.y))) * x.y + b.y;
        float v2 = (1.0f / (1.0f + expf(-x.z))) * x.z + b.z;
        float v3 = (1.0f / (1.0f + expf(-x.w))) * x.w + b.w;
        
        // Accumulate
        local_sum += v0 + v1 + v2 + v3;
        local_sq_sum += v0*v0 + v1*v1 + v2*v2 + v3*v3;
        
        // Store intermediate values
        shared_data[i * 4] = v0;
        shared_data[i * 4 + 1] = v1;
        shared_data[i * 4 + 2] = v2;
        shared_data[i * 4 + 3] = v3;
    }
    
    // Handle remainder
    for (int i = remainder_start + threadIdx.x; i < channels_per_group; i += blockDim.x) {
        int idx = base_offset + i;
        int feature_idx = group_start + i;
        
        float x = input[idx];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        float val = sigmoid_x * x + bias[feature_idx];
        
        local_sum += val;
        local_sq_sum += val * val;
        shared_data[i] = val;
    }
    
    // Store local sums to shared memory for reduction
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
    
    // Compute mean and variance (all threads read from shared memory)
    float mean = shared_sum[0] / channels_per_group;
    float variance = shared_sq_sum[0] / channels_per_group - mean * mean;
    float inv_std = rsqrtf(variance + eps);
    
    __syncthreads();
    
    // Second pass: normalize and apply gamma/beta using vectorized stores
    float4* output_vec = reinterpret_cast<float4*>(output + base_offset);
    const float4* gamma_vec = reinterpret_cast<const float4*>(gamma + group_start);
    const float4* beta_vec = reinterpret_cast<const float4*>(beta + group_start);
    
    for (int i = threadIdx.x; i < num_vec4; i += blockDim.x) {
        float4 g = gamma_vec[i];
        float4 bt = beta_vec[i];
        
        float v0 = shared_data[i * 4];
        float v1 = shared_data[i * 4 + 1];
        float v2 = shared_data[i * 4 + 2];
        float v3 = shared_data[i * 4 + 3];
        
        float4 result;
        result.x = ((v0 - mean) * inv_std) * g.x + bt.x;
        result.y = ((v1 - mean) * inv_std) * g.y + bt.y;
        result.z = ((v2 - mean) * inv_std) * g.z + bt.z;
        result.w = ((v3 - mean) * inv_std) * g.w + bt.w;
        
        output_vec[i] = result;
    }
    
    // Handle remainder
    for (int i = remainder_start + threadIdx.x; i < channels_per_group; i += blockDim.x) {
        int idx = base_offset + i;
        int feature_idx = group_start + i;
        
        float val = shared_data[i];
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
    int block_size = 64; // Optimized for channels_per_group = 64
    
    // Shared memory: sum, sq_sum, and intermediate data
    size_t shared_mem_size = (2 * block_size + channels_per_group) * sizeof(float);
    
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
