import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

fused_activation_norm_cpp_source = """
#include <hip/hip_runtime.h>

// Warp shuffle reduction helpers
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

__device__ float warpReduceSumShared(float val, float* shared, int tid) {
    int lane = tid % 32;
    int wid = tid / 32;
    
    val = warpReduceSum(val);
    
    if (lane == 0) {
        shared[wid] = val;
    }
    
    __syncthreads();
    
    val = (tid < blockDim.x / 32) ? shared[lane] : 0.0f;
    
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    
    return val;
}

__constant__ float eps = 1e-5f;

__global__ void fused_swish_bias_groupnorm_kernel(
    const float* x,
    const float* weight,  // GroupNorm gamma
    const float* bias_gn, // GroupNorm beta
    const float* bias_add, // Bias to add before GroupNorm
    float* out,
    int batch_size,
    int num_features,
    int num_groups
) {
    int batch_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    int total_threads = blockDim.x;
    
    int features_per_group = num_features / num_groups;
    
    // Shared memory for reduction and storing sums
    extern __shared__ float shared_data[];
    float* s_reduce = shared_data;
    float* s_sum = &shared_data[total_threads];  // For storing group sums
    float* s_sq_sum = &shared_data[total_threads + num_groups];
    
    // Initialize group sums
    if (thread_idx < num_groups) {
        s_sum[thread_idx] = 0.0f;
        s_sq_sum[thread_idx] = 0.0f;
    }
    __syncthreads();
    
    // Compute swish + bias and accumulate per-thread values
    for (int g = 0; g < num_groups; g++) {
        float thread_sum = 0.0f;
        float thread_sq_sum = 0.0f;
        int features_processed = 0;
        
        // Each thread works on strided features in this group
        for (int f = thread_idx; f < features_per_group; f += total_threads) {
            int feature_idx = g * features_per_group + f;
            int idx = batch_idx * num_features + feature_idx;
            
            float val = x[idx];
            
            // Swish: sigmoid(x) * x
            float sigmoid_val = 1.0f / (1.0f + expf(-val));
            float swish_val = sigmoid_val * val;
            
            // Add bias
            float b_val = bias_add[feature_idx];
            val = swish_val + b_val;
            
            // Accumulate
            thread_sum += val;
            thread_sq_sum += val * val;
            features_processed++;
        }
        
        // Reduce to get group statistics
        __syncthreads();
        float group_sum = warpReduceSumShared(thread_sum, s_reduce, thread_idx);
        float group_sq_sum = warpReduceSumShared(thread_sq_sum, s_reduce, thread_idx);
        
        if (thread_idx == 0) {
            s_sum[g] = group_sum;
            s_sq_sum[g] = group_sq_sum;
        }
        __syncthreads();
        
        float mean = s_sum[g] / features_per_group;
        float variance = s_sq_sum[g] / features_per_group - mean * mean;
        float inv_std = rsqrtf(variance + eps);
        
        // Apply normalization
        for (int f = thread_idx; f < features_per_group; f += total_threads) {
            int feature_idx = g * features_per_group + f;
            int idx = batch_idx * num_features + feature_idx;
            
            float val = x[idx];
            
            // Swish
            float sigmoid_val = 1.0f / (1.0f + expf(-val));
            float swish_val = sigmoid_val * val;
            
            // Add bias
            float b_val = bias_add[feature_idx];
            val = swish_val + b_val;
            
            // Normalize
            val = (val - mean) * inv_std;
            
            // Apply GroupNorm affine transform
            if (weight != nullptr) {
                val = val * weight[feature_idx];
            }
            if (bias_gn != nullptr) {
                val = val + bias_gn[feature_idx];
            }
            
            out[idx] = val;
        }
    }
}

torch::Tensor fused_swish_bias_groupnorm_hip(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias_gn,
    torch::Tensor bias_add,
    int num_groups
) {
    auto batch_size = x.size(0);
    auto num_features = x.size(1);
    
    auto out = torch::zeros_like(x);
    
    const int block_size = 256;
    const int num_blocks = batch_size;
    
    int shared_mem_size = (block_size + 2 * num_groups) * sizeof(float);
    
    fused_swish_bias_groupnorm_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_gn.data_ptr<float>(),
        bias_add.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        num_features,
        num_groups
    );
    
    return out;
}
"""

fused_swish_bias_groupnorm = load_inline(
    name="fused_swish_bias_groupnorm",
    cpp_sources=fused_activation_norm_cpp_source,
    functions=["fused_swish_bias_groupnorm_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model with fused Swish + Bias + GroupNorm kernel
    Uses warp-based reductions for better performance
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        
        # Use nn.Linear and nn.GroupNorm for proper initialization
        self._linear = nn.Linear(in_features, out_features)
        self._group_norm = nn.GroupNorm(num_groups, out_features)
        
        # Get the bias parameter (same as reference)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Store components for our kernel
        self.num_groups = num_groups
        
        # Fused kernel
        self.fused_kernel = fused_swish_bias_groupnorm
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Matrix multiplication
        x = torch.matmul(x, self._linear.weight.t()) + self._linear.bias.unsqueeze(0)
        
        # Fused: Swish + bias addition + GroupNorm
        x = self.fused_kernel.fused_swish_bias_groupnorm_hip(
            x, 
            self._group_norm.weight, 
            self._group_norm.bias, 
            self.bias, 
            self.num_groups
        )
        
        return x