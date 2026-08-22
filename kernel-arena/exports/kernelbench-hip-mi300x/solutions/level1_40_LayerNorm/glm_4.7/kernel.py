import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

layernorm_cpp_source = """
#include <hip/hip_runtime.h>
#include <math.h>

__global__ void layernorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int B, int C, int H, int W) {
    
    int b = blockIdx.x;
    int idx = b * C * H * W;  // Starting index for this batch
    
    // Compute mean and variance over entire batch element (C*H*W values)
    
    // Use shared memory for reductions
    extern __shared__ float shared_data[];
    
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    // Each thread processes multiple elements
    int tid = threadIdx.x;
    int total_elements = C * H * W;
    int stride = blockDim.x;
    
    for (int i = tid; i < total_elements; i += stride) {
        float val = x[idx + i];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    // Store in shared memory
    shared_data[tid] = local_sum;
    shared_data[tid + blockDim.x] = local_sum_sq;
    __syncthreads();
    
    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
            shared_data[tid + blockDim.x] += shared_data[tid + blockDim.x + s];
        }
        __syncthreads();
    }
    
    // Compute mean and variance
    float mean_val = shared_data[0] / (float)total_elements;
    float var_val = (shared_data[blockDim.x] / (float)total_elements) - (mean_val * mean_val);
    var_val = fmaxf(var_val, 0.0f);  // Ensure non-negative
    float inv_std = rsqrtf(var_val + 1e-5f);  // LayerNorm epsilon
    
    // Apply normalization with per-channel gamma and beta
    for (int i = tid; i < total_elements; i += stride) {
        int c_idx = i / (H * W);  // Channel index
        float val = x[idx + i];
        float norm_val = (val - mean_val) * inv_std;
        out[idx + i] = gamma[c_idx] * norm_val + beta[c_idx];
    }
}

torch::Tensor layernorm_hip(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta) {
    auto B = x.size(0);  // batch_size
    auto C = x.size(1);  // features/channels
    auto H = x.size(2);
    auto W = x.size(3);
    
    auto out = torch::zeros_like(x);
    
    int threads = 256;
    
    dim3 grid(B);
    dim3 block(threads);
    
    // 2 * threads for storing sum and sum_sq
    layernorm_kernel<<<grid, block, 2 * threads * sizeof(float)>>>(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        out.data_ptr<float>(),
        B, C, H, W
    );
    
    return out;
}
"""

layernorm_hip = load_inline(
    name="layernorm_hip",
    cpp_sources=layernorm_cpp_source,
    functions=["layernorm_hip"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized LayerNorm with custom HIP kernel.
    """
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        
        # Initialize learnable parameters
        self.gamma = nn.Parameter(torch.ones(normalized_shape[0]))  # weight
        self.beta = nn.Parameter(torch.zeros(normalized_shape[0]))  # bias
        
        # Load custom HIP kernel
        self.layernorm_hip = layernorm_hip
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Layer Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied.
        """
        return self.layernorm_hip.layernorm_hip(x, self.gamma, self.beta)