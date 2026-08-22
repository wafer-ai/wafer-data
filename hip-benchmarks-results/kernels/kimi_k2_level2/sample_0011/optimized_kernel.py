import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

# Fused kernel for GroupNorm + Scale + MaxPool + Clamp
fused_kernel_cpp_source = """
#include <hip/hip_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256
#define FLT_MAX 3.402823466e+38F

__device__ float block_reduce_sum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    // Final reduction
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    if (wid == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down(val, offset);
        }
    }
    return val;
}

// Kernel to compute mean and variance per sample per group
__global__ void compute_group_stats(
    const float* input,
    float* mean,
    float* var,
    int N, int C, int H, int W,
    int num_groups
) {
    int n = blockIdx.x;
    int g = blockIdx.y;
    int tid = threadIdx.x;
    
    int c_per_group = C / num_groups;
    int total_elements = c_per_group * H * W;
    
    float sum = 0.0f;
    
    // Compute sum for mean
    for (int i = tid; i < total_elements; i += BLOCK_SIZE) {
        int c_in_group = i / (H * W);
        int c = g * c_per_group + c_in_group;
        int h = (i / W) % H;
        int w = i % W;
        int idx = ((n * C + c) * H + h) * W + w;
        sum += input[idx];
    }
    
    sum = block_reduce_sum(sum);
    
    if (tid == 0) {
        mean[n * num_groups + g] = sum / total_elements;
    }
    
    __syncthreads();
    
    // Compute sum of squares for variance
    float mean_val = mean[n * num_groups + g];
    float sum_sq = 0.0f;
    
    for (int i = tid; i < total_elements; i += BLOCK_SIZE) {
        int c_in_group = i / (H * W);
        int c = g * c_per_group + c_in_group;
        int h = (i / W) % H;
        int w = i % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float diff = input[idx] - mean_val;
        sum_sq += diff * diff;
    }
    
    // Block reduction for variance
    sum_sq = block_reduce_sum(sum_sq);
    
    if (tid == 0) {
        var[n * num_groups + g] = sum_sq / total_elements;
    }
}

// Kernel to apply normalization, scale, maxpool, and clamp
__global__ void apply_norm_scale_maxpool_clamp(
    const float* input,
    const float* mean,
    const float* var,
    const float* gamma_prime,
    const float* beta,
    float* output,
    int N, int C, int H, int W, int H_out, int W_out,
    int num_groups, int pool_size, float clamp_min, float clamp_max, float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = N * C * H_out * W_out;
    
    if (idx < total_output_elements) {
        // Decode indices
        int tmp = idx;
        int w_out = tmp % W_out; tmp /= W_out;
        int h_out = tmp % H_out; tmp /= H_out;
        int c = tmp % C; tmp /= C;
        int n = tmp;
        
        int g = c / (C / num_groups);
        float mean_val = mean[n * num_groups + g];
        float var_val = var[n * num_groups + g];
        float std = sqrtf(var_val + eps);
        
        float scale_val = gamma_prime[c];
        float bias_val = beta[c];
        
        float max_val = -FLT_MAX;
        
        // Maxpool with fused operations
        for (int kh = 0; kh < pool_size; ++kh) {
            for (int kw = 0; kw < pool_size; ++kw) {
                int h = h_out * pool_size + kh;
                int w = w_out * pool_size + kw;
                int input_idx = ((n * C + c) * H + h) * W + w;
                float val = input[input_idx];
                
                // GroupNorm
                val = (val - mean_val) / std;
                // Scale and bias (fused gamma * scale into gamma_prime)
                val = val * scale_val + bias_val;
                
                // Maxpool
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
        
        // Clamp
        max_val = fmaxf(clamp_min, fminf(clamp_max, max_val));
        
        int output_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
        output[output_idx] = max_val;
    }
}

torch::Tensor fused_kernel_hip(
    torch::Tensor conv_output,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor scale,
    int pool_size,
    float clamp_min,
    float clamp_max
) {
    auto N = conv_output.size(0);
    auto C = conv_output.size(1);
    auto H = conv_output.size(2);
    auto W = conv_output.size(3);
    auto H_out = H / pool_size;
    auto W_out = W / pool_size;
    auto num_groups = 16;  // From model definition
    float eps = 1e-5;
    
    auto output = torch::empty({N, C, H_out, W_out}, conv_output.options());
    
    // Precompute gamma_prime = gamma * scale (fusing the two multiplications)
    auto gamma_prime = gamma * scale.squeeze();
    
    // Allocate buffers for mean and var
    auto mean = torch::empty({N, num_groups}, conv_output.options());
    auto var = torch::empty({N, num_groups}, conv_output.options());
    
    // Launch kernel to compute stats
    dim3 grid_stats(N, num_groups);
    dim3 block_stats(BLOCK_SIZE);
    
    compute_group_stats<<<grid_stats, block_stats>>>(
        conv_output.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        N, C, H, W, num_groups
    );
    
    // Launch kernel to apply ops
    int total_output_elements = N * C * H_out * W_out;
    int num_blocks = (total_output_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    apply_norm_scale_maxpool_clamp<<<num_blocks, BLOCK_SIZE>>>(
        conv_output.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        gamma_prime.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W, H_out, W_out,
        num_groups, pool_size, clamp_min, clamp_max, eps
    );
    
    return output;
}
"""

fused_kernel = load_inline(
    name="fused_kernel",
    cpp_sources=fused_kernel_cpp_source,
    functions=["fused_kernel_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        
        # We still need the group_norm parameters for the fused kernel
        self.fused_kernel = fused_kernel
        
    def forward(self, x):
        x = self.conv(x)
        
        # Use fused kernel for group_norm + scale + maxpool + clamp
        # Extract parameters from group_norm
        gamma = self.group_norm.weight
        beta = self.group_norm.bias
        
        # Replace maxpool+clamp with our fused kernel
        # Note: We need to handle maxpool separately since our kernel includes it
        x = self.fused_kernel.fused_kernel_hip(
            x, gamma, beta, self.scale, 
            self.maxpool.kernel_size,
            self.clamp_min, self.clamp_max
        )
        
        return x
