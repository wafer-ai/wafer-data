import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized GELU + Softmax fusion with shared memory
# Key optimization: compute GELU once per element, use shared memory for reduction
gelu_softmax_source = """
#include <hip/hip_runtime.h>

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void gelu_softmax_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int batch_size, int out_features) {
    
    // Process one row per block - use shared memory
    extern __shared__ float shared_gelu_vals[];
    
    int row = blockIdx.x;
    if (row >= batch_size) return;
    
    const float* row_ptr = x + row * out_features;
    float* out_ptr = out + row * out_features;
    
    int tid = threadIdx.x;
    
    // Each thread processes multiple elements for large rows
    int elements_per_thread = (out_features + blockDim.x - 1) / blockDim.x;
    int start = tid * elements_per_thread;
    int end = min(start + elements_per_thread, out_features);
    
    // Phase 1: Compute GELU values and store in shared memory
    for (int j = start; j < end; j++) {
        shared_gelu_vals[j] = gelu(row_ptr[j]);
    }
    __syncthreads();
    
    // Phase 2: Find maximum in shared memory using parallel reduction
    float max_val = -INFINITY;
    for (int j = tid; j < out_features; j += blockDim.x) {
        max_val = fmaxf(max_val, shared_gelu_vals[j]);
    }
    
    // Warp-level reduction
    __shared__ float shared_max[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    if (warp_id < 32 && lane_id == 0) {
        shared_max[warp_id] = max_val;
    }
    __syncthreads();
    
    if (tid == 0) {
        max_val = -INFINITY;
        for (int i = 0; i < min(32, blockDim.x); i++) {
            max_val = fmaxf(max_val, shared_max[i]);
        }
    }
    __syncthreads();
    
    // Broadcast max to all threads
    if (tid == 0) {
        shared_max[0] = max_val;
    }
    __syncthreads();
    max_val = shared_max[0];
    
    // Phase 3: Compute sum of exp(gelu - max)
    float sum = 0.0f;
    for (int j = tid; j < out_features; j += blockDim.x) {
        sum += expf(shared_gelu_vals[j] - max_val);
    }
    
    // Parallel reduction for sum
    __shared__ float shared_sum[32];
    if (warp_id < 32 && lane_id == 0) {
        shared_sum[warp_id] = sum;
    }
    __syncthreads();
    
    if (tid == 0) {
        sum = 0.0f;
        for (int i = 0; i < min(32, blockDim.x); i++) {
            sum += shared_sum[i];
        }
        shared_sum[0] = 1.0f / (sum + 1e-8f);
    }
    __syncthreads();
    
    float inv_sum = shared_sum[0];
    
    // Phase 4: Compute final softmax
    for (int j = start; j < end; j++) {
        out_ptr[j] = expf(shared_gelu_vals[j] - max_val) * inv_sum;
    }
}

torch::Tensor gelu_softmax(torch::Tensor x) {
    int batch_size = x.size(0);
    int out_features = x.size(1);
    
    auto out = torch::zeros_like(x);
    
    // Use 256 threads per block, and 1 block per row (or fewer if very large)
    int block_size = 256;
    int num_blocks = batch_size;
    
    // Shared memory: need space for one row of GELU values
    int shared_mem_size = out_features * sizeof(float);
    
    gelu_softmax_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size, out_features
    );
    
    return out;
}
"""

gelu_softmax = load_inline(
    name="gelu_softmax_large",
    cpp_sources=gelu_softmax_source,
    functions=["gelu_softmax"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies GELU, and then applies Softmax.
    Uses PyTorch's optimized matmul (rocBLAS) with fused GELU+Softmax using shared memory.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.gelu_softmax = gelu_softmax

    def forward(self, x):
        # Use PyTorch's optimized matmul, but use fused GELU+Softmax
        x = self.linear(x)
        x = self.gelu_softmax.gelu_softmax(x)
        return x