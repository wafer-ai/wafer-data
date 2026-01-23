import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized LayerNorm HIP kernel with better parallelism
layer_norm_hip_code = """
#include <hip/hip_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum(float val) {
    // Fast warp shuffle reduction (no shared memory)
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor(val, mask);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, int tid) {
    // Block reduction using warp shuffle across warp leaders
    __shared__ float shared_val[32];
    int lane = tid & 31;
    int warp_id = tid >> 5;
    
    // Warp-level reduction
    val = warp_reduce_sum(val);
    
    // Store warp results to shared
    if (lane == 0) {
        shared_val[warp_id] = val;
    }
    __syncthreads();
    
    // Final reduction in warp 0
    if (warp_id == 0) {
        val = (lane < (BLOCK_SIZE + 31) / 32) ? shared_val[lane] : 0;
        val = warp_reduce_sum(val);
    }
    
    return val;
}

__global__ void layer_norm_optimized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int normalized_size,
    float eps
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* batch_input = input + batch_idx * normalized_size;
    float* batch_output = output + batch_idx * normalized_size;
    
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int warps_in_block = (BLOCK_SIZE + 31) / 32;
    
    __shared__ float shared_vals[64];
    __shared__ float mean;
    __shared__ float rstd;
    __shared__ float warp_sums[32];
    
    // Phase 1: Parallel reduction for mean using warp shuffles
    float local_sum = 0.0f;
    
    // Grid-stride within block - each thread processes multiple elements
    for (int i = tid; i < normalized_size; i += BLOCK_SIZE) {
        local_sum += batch_input[i];
    }
    
    // Warp shuffle reduction
    float warp_sum = warp_reduce_sum(local_sum);
    
    // Store warp sums to shared memory
    if (lane == 0) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();
    
    // Block-level reduction (only first warp participates)
    if (warp_id == 0) {
        float block_sum = 0.0f;
        int active_warps = min(warps_in_block, (normalized_size + 31) / 32);
        
        for (int i = lane; i < active_warps; i += WARP_SIZE) {
            block_sum += warp_sums[i];
        }
        block_sum = warp_reduce_sum(block_sum);
        
        if (lane == 0) {
            mean = block_sum / normalized_size;
        }
    }
    __syncthreads();
    
    // Phase 2: Parallel reduction for variance
    float local_var_sum = 0.0f;
    float mean_val = mean;  // Load once from shared memory
    
    for (int i = tid; i < normalized_size; i += BLOCK_SIZE) {
        float diff = batch_input[i] - mean_val;
        local_var_sum += diff * diff;
    }
    
    // Same warp-shuffle reduction for variance
    warp_sum = warp_reduce_sum(local_var_sum);
    
    if (lane == 0) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float block_var_sum = 0.0f;
        int active_warps = min(warps_in_block, (normalized_size + 31) / 32);
        
        for (int i = lane; i < active_warps; i += WARP_SIZE) {
            block_var_sum += warp_sums[i];
        }
        block_var_sum = warp_reduce_sum(block_var_sum);
        
        if (lane == 0) {
            float variance = block_var_sum / normalized_size;
            rstd = rsqrtf(variance + eps);  // Reciprocal sqrt is faster
        }
    }
    __syncthreads();
    
    // Phase 3: Fused operations with better memory accesses
    float rstd_val = rstd;  // Load once from shared memory
    
    // Grid-stride within block - each thread processes multiple elements
    for (int i = tid; i < normalized_size; i += BLOCK_SIZE) {
        float normalized = (batch_input[i] - mean_val) * rstd_val;
        batch_output[i] = normalized * weight[i] + bias[i];
    }
}

torch::Tensor layer_norm_hip(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float eps) {
    int batch_size = 1;
    for (int i = 0; i < input.dim() - weight.dim(); i++) {
        batch_size *= input.size(i);
    }
    int normalized_size = weight.numel();
    
    auto output = torch::empty_like(input);
    
    dim3 blocks(batch_size);
    dim3 threads(BLOCK_SIZE);
    
    hipLaunchKernelGGL(
        layer_norm_optimized_kernel,
        blocks,
        threads,
        0, 0,
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        normalized_size,
        eps
    );
    
    return output;
}
"""

# Compile the HIP code
layer_norm_hip = load_inline(
    name='layer_norm_hip',
    cpp_sources=layer_norm_hip_code,
    functions=['layer_norm_hip'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(normalized_shape, dtype=torch.float32))
        self.eps = 1e-5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward to optimized HIP kernel
        output = layer_norm_hip.layer_norm_hip(x, self.weight, self.bias, self.eps)
        return output

def get_inputs():
    batch_size = 16
    features = 64
    dim1 = 256
    dim2 = 256
    x = torch.rand(batch_size, features, dim1, dim2, dtype=torch.float32).cuda()
    return [x]

def get_init_inputs():
    return [(64, 256, 256)]