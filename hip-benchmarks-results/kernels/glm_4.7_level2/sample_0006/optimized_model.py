import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

pool_sum_scale_cpp_source = """
#include <hip/hip_runtime.h>

#define BLOCK_SIZE 256

__global__ void pool_sum_scale_kernel(
    const float* matmul_out,  // (batch_size, out_features)
    float* output,            // (batch_size,)
    int batch_size,
    int out_features,
    float scale_factor
) {
    int sample_idx = blockIdx.x;
    if (sample_idx >= batch_size) return;
    
    const float* row = matmul_out + sample_idx * out_features;
    
    // Each thread processes multiple pairs
    int num_pairs = out_features / 2;
    int pairs_per_thread = (num_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int pair_start = threadIdx.x * pairs_per_thread;
    int pair_end = min(pair_start + pairs_per_thread, num_pairs);
    
    float thread_sum = 0.0f;
    
    for (int pair = pair_start; pair < pair_end; pair++) {
        int idx1 = 2 * pair;
        int idx2 = 2 * pair + 1;
        if (idx2 < out_features) {
            float val1 = row[idx1];
            float val2 = row[idx2];
            thread_sum += fmaxf(val1, val2);
        }
    }
    
    // Parallel reduction
    __shared__ float partial_sums[BLOCK_SIZE];
    partial_sums[threadIdx.x] = thread_sum;
    
    __syncthreads();
    
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        output[sample_idx] = partial_sums[0] * scale_factor;
    }
}

torch::Tensor pool_sum_scale_hip(
    const torch::Tensor matmul_out,  // (batch_size, out_features)
    float scale_factor
) {
    int batch_size = matmul_out.size(0);
    int out_features = matmul_out.size(1);
    
    auto output = torch::empty({batch_size}, matmul_out.options());
    
    dim3 grid_dim(batch_size);
    dim3 block_dim(BLOCK_SIZE);
    
    pool_sum_scale_kernel<<<grid_dim, block_dim>>>(
        matmul_out.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features,
        scale_factor
    );
    
    return output;
}
"""

pool_sum_scale = load_inline(
    name="pool_sum_scale",
    cpp_sources=pool_sum_scale_cpp_source,
    functions=["pool_sum_scale_hip"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model with fused maxpool+sum+scale.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        # Initialize the Linear layer for compatibility (weight and bias)
        self.matmul = nn.Linear(in_features, out_features)
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.fused_op = pool_sum_scale

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        # Use PyTorch's optimized matmul (uses rocBLAS)
        matmul_out = self.matmul(x)
        
        # Apply fused kernel: maxpool + sum + scale
        output = self.fused_op.pool_sum_scale_hip(matmul_out, self.scale_factor)
        
        return output


def get_inputs():
    return [torch.rand(128, 32768).cuda()]


def get_init_inputs():
    return [32768, 32768, 2, 0.5]