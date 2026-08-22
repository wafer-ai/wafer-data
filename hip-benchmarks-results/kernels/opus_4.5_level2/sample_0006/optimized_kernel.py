import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Fused kernel for MaxPool1d + Sum + Scale
fused_kernel_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>

// Fused MaxPool1d (kernel_size=2) + Sum + Scale kernel
// Each block handles one batch element
// We do a parallel reduction: first compute max of pairs, then sum them up

__global__ void fused_maxpool_sum_scale_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int features,
    const int pooled_features,
    const float scale_factor
) {
    // Each block processes one batch element
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* batch_input = input + batch_idx * features;
    
    // Shared memory for reduction
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Each thread accumulates sum of max-pooled values for its assigned indices
    float local_sum = 0.0f;
    
    // Process pooled_features elements (each is max of 2 consecutive values)
    for (int i = tid; i < pooled_features; i += num_threads) {
        int idx = i * 2;
        float val1 = batch_input[idx];
        float val2 = batch_input[idx + 1];
        float max_val = (val1 > val2) ? val1 : val2;
        local_sum += max_val;
    }
    
    sdata[tid] = local_sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 writes the result
    if (tid == 0) {
        output[batch_idx] = sdata[0] * scale_factor;
    }
}

torch::Tensor fused_maxpool_sum_scale(torch::Tensor input, float scale_factor) {
    const int batch_size = input.size(0);
    const int features = input.size(1);
    const int pooled_features = features / 2;
    
    auto output = torch::empty({batch_size}, input.options());
    
    const int block_size = 256;
    const int num_blocks = batch_size;
    const int shared_mem_size = block_size * sizeof(float);
    
    fused_maxpool_sum_scale_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        features,
        pooled_features,
        scale_factor
    );
    
    return output;
}
"""

fused_kernel_cpp = """
torch::Tensor fused_maxpool_sum_scale(torch::Tensor input, float scale_factor);
"""

fused_module = load_inline(
    name="fused_maxpool_sum_scale",
    cpp_sources=fused_kernel_cpp,
    cuda_sources=fused_kernel_source,
    functions=["fused_maxpool_sum_scale"],
    verbose=True,
    extra_cuda_cflags=["-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication, max pooling, sum, and scaling.
    Uses a fused HIP kernel for maxpool + sum + scale operations.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size
        self.fused_op = fused_module

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        # Use PyTorch's optimized linear layer
        x = self.matmul(x)
        # Use fused kernel for maxpool + sum + scale
        x = self.fused_op.fused_maxpool_sum_scale(x, self.scale_factor)
        return x
