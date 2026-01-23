import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

layernorm_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cmath>

// Warp reduce sum using warp shuffles
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 32; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Block reduce sum using shared memory
__device__ __forceinline__ float blockReduceSum(float val, float* shared) {
    int lane = threadIdx.x % 64;
    int wid = threadIdx.x / 64;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Read from shared memory only if that warp existed
    int numWarps = (blockDim.x + 63) / 64;
    val = (threadIdx.x < numWarps) ? shared[threadIdx.x] : 0.0f;

    if (wid == 0) val = warpReduceSum(val);
    return val;
}

// LayerNorm kernel: each block processes one batch element
// normalized_size is the number of elements to normalize over
__global__ void layernorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int batch_size,
    int normalized_size,
    float eps
) {
    __shared__ float shared_mem[32]; // For block reduction
    __shared__ float mean_shared;
    __shared__ float inv_std_shared;

    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const float* x = input + batch_idx * normalized_size;
    float* y = output + batch_idx * normalized_size;

    // Step 1: Compute mean using parallel reduction
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        local_sum += x[i];
    }
    
    float total_sum = blockReduceSum(local_sum, shared_mem);
    
    if (threadIdx.x == 0) {
        mean_shared = total_sum / (float)normalized_size;
    }
    __syncthreads();
    
    float mean = mean_shared;

    // Step 2: Compute variance using parallel reduction
    float local_var_sum = 0.0f;
    for (int i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        float diff = x[i] - mean;
        local_var_sum += diff * diff;
    }
    
    float total_var = blockReduceSum(local_var_sum, shared_mem);
    
    if (threadIdx.x == 0) {
        float variance = total_var / (float)normalized_size;
        inv_std_shared = rsqrtf(variance + eps);
    }
    __syncthreads();
    
    float inv_std = inv_std_shared;

    // Step 3: Normalize and apply affine transformation
    for (int i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        float normalized = (x[i] - mean) * inv_std;
        y[i] = normalized * gamma[i] + beta[i];
    }
}

torch::Tensor layernorm_hip(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, double eps) {
    auto batch_size = input.size(0);
    auto normalized_size = input.numel() / batch_size;
    
    auto output = torch::empty_like(input);
    
    // Use 1024 threads per block for good occupancy
    const int block_size = 1024;
    const int num_blocks = batch_size;
    
    layernorm_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        normalized_size,
        (float)eps
    );
    
    return output;
}
"""

layernorm_cpp_source = """
torch::Tensor layernorm_hip(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, double eps);
"""

layernorm_module = load_inline(
    name="layernorm_hip",
    cpp_sources=layernorm_cpp_source,
    cuda_sources=layernorm_hip_source,
    functions=["layernorm_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-ffast-math"]
)


class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization using custom HIP kernel.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = 1e-5
        
        # Initialize learnable parameters
        normalized_size = 1
        for s in normalized_shape:
            normalized_size *= s
        
        self.weight = nn.Parameter(torch.ones(normalized_size))
        self.bias = nn.Parameter(torch.zeros(normalized_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        # Ensure contiguous
        x = x.contiguous()
        return layernorm_module.layernorm_hip(x, self.weight, self.bias, self.eps)


def get_inputs():
    x = torch.rand(16, 64, 256, 256).cuda()
    return [x]


def get_init_inputs():
    return [(64, 256, 256)]
