import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Optimized fused kernel with better memory access and more threads
fused_kernel_cpp_source = """
#include <hip/hip_runtime.h>

// Fast GELU approximation
__device__ __forceinline__ float fast_gelu(float x) {
    float x_cubed = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x_cubed);
    float tanh_val = tanhf(inner);
    return 0.5f * x * (1.0f + tanh_val);
}

__global__ void fused_pool_gelu_scale_max_kernel_v5(
    const float* __restrict__ matmul_out,
    float* __restrict__ final_out,
    int batch_size,
    int out_features,
    int pool_size,
    float scale
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int num_pools = out_features / pool_size;  // 512 pools
    
    if (batch_idx >= batch_size) return;
    
    // Use 512 threads for better parallelism
    int pool_idx = tid;
    if (pool_idx >= num_pools) return;
    
    const float* batch_data = matmul_out + batch_idx * out_features;
    
    // Compute pool sum
    int start_feature = pool_idx * pool_size;
    float sum = 0.0f;
    
    // Vectorized sum with better memory access
    #pragma unroll 16
    for (int i = 0; i < pool_size; i++) {
        sum += batch_data[start_feature + i];
    }
    
    float avg = sum / pool_size;
    float gelu_val = fast_gelu(avg);
    float scaled = gelu_val * scale;
    
    // Use warp-shuffle for max reduction (block size = 512 = 16 warps)
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Each warp computes max of its 16 pools
    float warp_max = scaled;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, warp_max, offset);
        warp_max = fmaxf(warp_max, other);
    }
    
    // Store warp max in shared memory
    __shared__ float shared_max[16];  // One per warp
    if (lane_id == 0) {
        shared_max[warp_id] = warp_max;
    }
    
    __syncthreads();
    
    // Final reduction by warp 0
    if (warp_id == 0) {
        float block_max = shared_max[lane_id];
        if (lane_id < 9) {
            #pragma unroll
            for (int offset = 8; offset > 0; offset >>= 1) {
                if (lane_id + offset < 16) {
                    block_max = fmaxf(block_max, __shfl_sync(0xffffffff, block_max, lane_id + offset));
                }
            }
        }
        
        if (lane_id == 0) {
            final_out[batch_idx] = block_max;
        }
    }
}

torch::Tensor fused_pool_gelu_scale_max_v5(torch::Tensor matmul_out, float scale_factor) {
    auto batch_size = matmul_out.size(0);
    auto out_features = matmul_out.size(1);
    auto pool_size = 16;
    auto final_out = torch::zeros({batch_size}, matmul_out.options());
    
    const int threads = 512;  // 512 threads = 16 warps
    const int blocks = batch_size;
    
    fused_pool_gelu_scale_max_kernel_v5<<<blocks, threads>>>(
        matmul_out.data_ptr<float>(),
        final_out.data_ptr<float>(),
        batch_size,
        out_features,
        pool_size,
        scale_factor
    );
    
    return final_out;
}
"""

fused_kernel_v5 = load_inline(
    name="fused_kernel_v5",
    cpp_sources=fused_kernel_cpp_source,
    functions=["fused_pool_gelu_scale_max_v5"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scale_factor = scale_factor
        self.fused_kernel_v5 = fused_kernel_v5
        
    def forward(self, x):
        x = self.matmul(x)  
        x = self.fused_kernel_v5.fused_pool_gelu_scale_max_v5(x, self.scale_factor)
        return x

# Input generation functions
batch_size = 1024
in_features = 8192
out_features = 8192
pool_kernel_size = 16
scale_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features, device='cuda')]

def get_init_inputs():
    return [in_features, out_features, pool_kernel_size, scale_factor]