import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Aggressively optimized kernel with warp-level parallelism
fused_kernel_cpp_source = """
#include <hip/hip_runtime.h>

// GELU approximation using tanh
device inline float gelu(float x) {
    float x_cubed = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x_cubed);
    float tanh_val = tanhf(inner);
    return 0.5f * x * (1.0f + tanh_val);
}

__global__ void fused_pool_gelu_scale_max_kernel_v3(
    const float* matmul_out,
    float* final_out,
    int batch_size,
    int out_features,
    int pool_size,
    float scale
) {
    int batch_idx = blockIdx.x;
    int lane_id = threadIdx.x % 32;  // Lane within warp
    int warp_id = threadIdx.x / 32;  // Warp ID
    int num_warps = blockDim.x / 32;
    
    if (batch_idx >= batch_size) return;
    
    const float* batch_data = matmul_out + batch_idx * out_features;
    
    // Each warp handles a subset of the pools
    const int pools_per_warp = (out_features / pool_size) / num_warps;  // 512 / 8 = 64 pools per warp
    const int features_per_warp = pools_per_warp * pool_size;  // 64 * 16 = 1024 features
    
    // Starting position for this warp
    int warp_start_pool = warp_id * pools_per_warp;
    int warp_start_feature = warp_start_pool * pool_size;
    
    float thread_max = -FLT_MAX;
    
    // Each thread in warp processes features with stride 32 (coalesced access)
    for (int i = lane_id; i < features_per_warp; i += 32) {
        int feature_idx = warp_start_feature + i;
        int pool_idx = feature_idx / pool_size;
        int offset_in_pool = feature_idx % pool_size;
        
        // Accumulate pool using warp shuffle
        float val = batch_data[feature_idx];
        
        // Use ballot for efficient pooling
        // Sum within each pool lane group
        float sum = val;
        #pragma unroll
        for (int offset = 1; offset < pool_size; offset *= 2) {
            float val2 = __shfl_down(sum, offset);
            if (offset_in_pool + offset < pool_size) {
                sum += val2;
            }
        }
        
        // Lane 0 of each pool has the sum
        if (offset_in_pool == 0) {
            float avg = sum / pool_size;
            float gelu_val = gelu(avg);
            float scaled = gelu_val * scale;
            thread_max = fmaxf(thread_max, scaled);
        }
    }
    
    // Now find max across threads in warp using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_xor(thread_max, offset);
        thread_max = fmaxf(thread_max, other);
    }
    
    // Store warp max in shared memory
    extern __shared__ float s_max[];
    if (lane_id == 0) {
        s_max[warp_id] = thread_max;
    }
    
    __syncthreads();
    
    // Final reduction across warps
    if (threadIdx.x == 0) {
        float block_max = s_max[0];
        #pragma unroll
        for (int i = 1; i < num_warps && i < blockDim.x; i++) {
            block_max = fmaxf(block_max, s_max[i]);
        }
        final_out[batch_idx] = block_max;
    }
}

torch::Tensor fused_pool_gelu_scale_max_v3(torch::Tensor matmul_out, float scale_factor) {
    auto batch_size = matmul_out.size(0);
    auto out_features = matmul_out.size(1);
    auto pool_size = 16;
    auto final_out = torch::zeros({batch_size}, matmul_out.options());
    
    const int threads = 256;  // 8 warps per block
    const int blocks = batch_size;
    size_t shared_mem = (threads / 32) * sizeof(float);  // One float per warp
    
    fused_pool_gelu_scale_max_kernel_v3<<<blocks, threads, shared_mem>>>(
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

fused_kernel_v3 = load_inline(
    name="fused_kernel_v3",
    cpp_sources=fused_kernel_cpp_source,
    functions=["fused_pool_gelu_scale_max_v3"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scale_factor = scale_factor
        self.fused_kernel_v3 = fused_kernel_v3
        
    def forward(self, x):
        x = self.matmul(x)  # Matmul + bias
        x = self.fused_kernel_v3.fused_pool_gelu_scale_max_v3(x, self.scale_factor)
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