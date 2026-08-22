import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Custom fused kernel: avg_pool + gelu + scale + max
fused_kernel_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void fused_pool_gelu_scale_max_kernel(
    const float* matmul_out,
    float* final_out,
    int batch_size,
    int out_features,
    int pool_size,
    float scale
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    if (batch_idx >= batch_size) return;
    
    const float* batch_matmul = matmul_out + batch_idx * out_features;
    __shared__ float pooled[512];
    
    const int features_per_thread = out_features / block_size; // 8192 / 256 = 32
    const int pool_groups_per_thread = features_per_thread / pool_size; // 32 / 16 = 2
    
    // Local sums for the pool groups this thread handles
    float sum[2] = {0.0f, 0.0f};
    
    // Accumulate features into pool groups
    #pragma unroll
    for (int i = 0; i < features_per_thread; i++) {
        int feature_idx = tid * features_per_thread + i;
        int pool_idx = feature_idx / pool_size;
        int local_pool_idx = pool_idx - (tid * pool_groups_per_thread);
        
        float val = batch_matmul[feature_idx];
        sum[local_pool_idx] += val;
    }
    
    // Compute pooled values: average, GELU, scale
    #pragma unroll
    for (int p = 0; p < pool_groups_per_thread; p++) {
        int pool_idx = tid * pool_groups_per_thread + p;
        
        float avg = sum[p] / pool_size;
        
        // GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float x_cubed = avg * avg * avg;
        float inner = 0.7978845608f * (avg + 0.044715f * x_cubed);
        float tanh_val = tanhf(inner);
        float gelu_val = 0.5f * avg * (1.0f + tanh_val);
        
        float scaled = gelu_val * scale;
        pooled[pool_idx] = scaled;
    }
    
    __syncthreads();
    
    // Parallel reduction to find max (512 elements)
    if (tid < 256) {
        pooled[tid] = fmaxf(pooled[tid], pooled[tid + 256]);
    }
    __syncthreads();
    
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            pooled[tid] = fmaxf(pooled[tid], pooled[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        final_out[batch_idx] = pooled[0];
    }
}

torch::Tensor fused_pool_gelu_scale_max(torch::Tensor matmul_out, float scale_factor) {
    auto batch_size = matmul_out.size(0);
    auto out_features = matmul_out.size(1);
    auto pool_size = 16;
    auto final_out = torch::zeros({batch_size}, matmul_out.options());
    
    const int threads = 256;
    const int blocks = batch_size;
    
    fused_pool_gelu_scale_max_kernel<<<blocks, threads>>>(
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

fused_kernel = load_inline(
    name="fused_kernel",
    cpp_sources=fused_kernel_cpp_source,
    functions=["fused_pool_gelu_scale_max"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scale_factor = scale_factor
        self.fused_kernel = fused_kernel
        
    def forward(self, x):
        x = self.matmul(x)  # Matmul + bias
        x = self.fused_kernel.fused_pool_gelu_scale_max(x, self.scale_factor)
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